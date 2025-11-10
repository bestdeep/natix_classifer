# train_combined.py
import os
import argparse
import random
from datetime import datetime
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# user project imports (assumed present)
from model.backbone import get_backbone_builder
from datasets.natix_dataset import NatixDataset, MaxOversamplingBatchSampler
from model.utils import (
    set_seed,
    make_transforms,
    mixup_data,
    cutmix_data,
    mixup_criterion,
    collate_fn,
    collate_fn_multi,
    evaluate,
    ensure_tensor_labels,
    infer_domain_from_path,
    FocalLoss,
)

# -------------------------
# Single model training (supervised)
# -------------------------
def train_single_model(args, model_name: str, train_ds, val_ds, device: torch.device):
    print(f"\n=== Training model: {model_name} ===")
    num_classes = len(train_ds.label_map) if hasattr(train_ds, "label_map") else len(train_ds.dataset.label_map)
    model = None

    # use simple backbone builder
    model_builder = get_backbone_builder(model_name)
    try:
        model, _ = model_builder(num_classes=num_classes, pretrained=args.pretrained)
    except Exception as e:
        print(f"Warning: failed to build {model_name} pretrained; fallback to pretrained=False ({e})")
        model, _ = model_builder(num_classes=num_classes, pretrained=False)

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_counts = Counter(train_ds.labels)
    print(f"[INFO] Class distribution in training set:")
    for cls, count in sorted(class_counts.items()):
        print(f" - Class {cls}: {count} samples")

    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(num_classes)]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=weights_tensor, gamma=2.0, reduction="mean")

    scaler = GradScaler("cuda")

    # DataLoaders: Subset objects may be passed (your main script uses Subset)
    train_sampler = MaxOversamplingBatchSampler(train_ds.real_indices, train_ds.synthetic_indices, batch_size=args.batch_size)
    val_sampler = MaxOversamplingBatchSampler(val_ds.real_indices, val_ds.synthetic_indices, batch_size=args.batch_size)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn_multi, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate_fn_multi, num_workers=args.num_workers, pin_memory=True)

    # TensorBoard
    writer = None
    if args.tb_logdir:
        os.makedirs(args.tb_logdir, exist_ok=True)
        tb_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, tb_name))
        print("TensorBoard logs ->", os.path.join(args.tb_logdir, tb_name))

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output, model_name + "_best.pth")
    os.makedirs(args.output, exist_ok=True)

    # Resume if exists
    if args.resume and os.path.isfile(best_ckpt_path):
        print(f"Resuming training from checkpoint: {best_ckpt_path}")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state", {}))

    global_step = 0
    # ---------- Supervised training (full supervised) ----------
    print("Starting supervised training for", args.epochs, "epochs")
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        running_loss, correct, total = 0.0, 0, 0
        for step, batch in enumerate(loop):
            images_stack, labels, _, _ = batch
            # images_stack: (B, K, C, H, W)
            B, K, C, H, W = images_stack.shape
            imgs_flat = images_stack.view(B * K, C, H, W).to(device)   # (B*K, C, H, W)
            labels = ensure_tensor_labels(labels).to(device)
            labels_flat = labels.unsqueeze(1).repeat(1, K).view(-1).to(device)  # (B*K,)
            mask = labels_flat >= 0

            imgs, labels = imgs_flat[mask], labels_flat[mask]

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                out = model(imgs)
                if isinstance(out, tuple):
                    class_logits = out[0]
                else:
                    class_logits = out
                loss = criterion(class_logits, labels)                    

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(class_logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if writer and (step % args.log_every_steps == 0):
                writer.add_scalar("train/step_loss", float(loss.item()), global_step)

            global_step += 1
            acc = 100.0 * correct / total if total > 0 else 0.0
            loop.set_postfix(loss=float(loss.item()), acc=acc)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        scheduler.step()

        # validation
        val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)        
        # tensorboard logging
        if writer:
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("train/epoch_acc", train_acc, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    writer.add_scalar(f"val/{k}", v, epoch)
        # checkpointing
        # save last
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, os.path.join(args.output, model_name + "_last.pth"))

        total_acc = (val_metrics["accuracy"] + val_metrics["mcc"]) / 2.0
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f} Total Acc: {total_acc:.4f}")

        if total_acc > best_val_acc:
            best_val_acc = total_acc
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, best_ckpt_path)
            print("Saved new best checkpoint ->", best_ckpt_path)
            no_progress = 0
        else:
            no_progress += 1

        if no_progress >= 2:
            print("No progress in validation for 2 epochs, stopping training early.")
            break

    if writer:
        writer.close()
    print(f"Model {model_name} best checkpoint saved at: {best_ckpt_path}")
    return best_ckpt_path

# -------------------------
# Main entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Backbone model name to train and ensemble")
    parser.add_argument("--train-path", required=True, help="File list path for training images")
    parser.add_argument("--val-path", required=True, help="File list path for validation images")
    parser.add_argument("--output", type=str, default="checkpoints_real_synth", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--tb-logdir", type=str, default=None, help="TensorBoard root dir")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Enable resume from last checkpoint if exists")
    parser.add_argument("--synthetic-indicators", nargs="*", default=["synthetic", "synth", "ai_", "t2i", "i2i", "generated"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    print("Using device:", device)
    print("Training path:", args.train_path)
    print("Validation path:", args.val_path)
    print("Model to train:", args.model)

    # transforms
    transform_train = make_transforms(args.image_size)
    transform_val = make_transforms(args.image_size)

    # build dataset
    train_ds = NatixDataset(filelist_path=args.train_path, transform=transform_train, augment=args.augment, num_augmentations=4)
    val_ds = NatixDataset(filelist_path=args.val_path, transform=transform_val, augment=args.augment, num_augmentations=4)

    num_samples = len(train_ds) + len(val_ds)

    print(f"Total samples: {num_samples}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    best_ckpts = []
    ckpt = train_single_model(args, args.model, train_ds, val_ds, device)
