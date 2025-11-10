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
from datasets.natix_dataset import NatixDataset, BalancedDomainSampler
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
from model.roadwalk import RoadworkClassifier


# -------------------------
# DANN training epoch
# -------------------------
def train_epoch_dann(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    scaler: GradScaler,
    global_step: int,
    total_steps: int,
    lambda_domain_weight: float,
    tb_writer: Optional[SummaryWriter],
    weights: Optional[torch.tensor] = None,
):
    model.train()
    criterion_cls = nn.CrossEntropyLoss(weight=weights, reduction="mean") if weights is not None else nn.CrossEntropyLoss(reduction="mean")
    criterion_domain = nn.CrossEntropyLoss(reduction="mean")
    running_loss = 0.0
    y_true, y_pred, y_scores = [], [], []
    pbar = tqdm(loader, desc=f"DANN-train", leave=False)
    total_samples = 0
    for step, batch in enumerate(pbar):
        images_stack, labels, _, domains = batch
        # images_stack: (B, K, C, H, W)
        B, K, C, H, W = images_stack.shape
        imgs = images_stack.view(B * K, C, H, W).to(device)   # (B*K, C, H, W)
        labels = ensure_tensor_labels(labels).to(device)
        labels = labels.unsqueeze(1).repeat(1, K).view(-1).to(device)  # (B*K,)
        mask = labels >= 0
        domain_labels = ensure_tensor_labels(domains).to(device)
        domain_labels = domain_labels.unsqueeze(1).repeat(1, K).view(-1).to(device)  # (B*K,)

        # dynamic GRL lambda schedule (DANN paper common schedule)
        p = min(1.0, float(global_step) / float(max(1, total_steps)))
        grl_lambda = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        # set grl if model supports it
        if hasattr(model, "set_grl_lambda"):
            model.set_grl_lambda(float(grl_lambda))

        # Mixup/CutMix still supported for DANN if desired (optional)
        # Here we keep simple: no MixUp/CutMix in DANN stage to avoid domain label confusion.
        with autocast(device_type="cuda"):
            out = model(imgs)
            # expect (class_logits, domain_logits) for DANN model
            if isinstance(out, tuple) and len(out) >= 2:
                class_logits, domain_logits = out[0], out[1]
            else:
                raise RuntimeError("DANN requires model returning (class_logits, domain_logits)")

            cls_loss = torch.tensor(0.0, device=device)
            if mask.sum() > 0:
                cls_loss = criterion_cls(class_logits[mask], labels[mask])
            domain_loss = criterion_domain(domain_logits, domain_labels)
            loss = cls_loss + lambda_domain_weight * domain_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if tb_writer and (global_step % 100 == 0):
            tb_writer.add_scalar("train/dann_grl_lambda", float(grl_lambda), global_step)
            tb_writer.add_scalar("train/dann_cls_loss", float(cls_loss.detach().item()) if isinstance(cls_loss, torch.Tensor) else float(cls_loss), global_step)
            tb_writer.add_scalar("train/dann_domain_loss", float(domain_loss.detach().item()) if isinstance(domain_loss, torch.Tensor) else float(domain_loss), global_step)
            tb_writer.add_scalar("train/dann_loss", float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss), global_step)

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        total_samples += labels.size(0)
        if mask.sum() > 0:
            probs = torch.softmax(class_logits[mask], dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(class_logits[mask], dim=1).detach().cpu().numpy().tolist()
            trues = labels[mask].cpu().numpy().tolist()
            y_scores.extend(probs.tolist())
            y_pred.extend(preds)
            y_true.extend(trues)

        global_step += 1
        pbar.set_postfix(loss=running_loss / total_samples)
    metrics = {}
    if len(y_true) > 0:
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        auc = None
        try:
            if len(set(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = None
        metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "auc": auc}
    metrics["loss"] = running_loss / total_samples
    return metrics, global_step

# -------------------------
# Single model training (supervised or DANN)
# -------------------------
def train_single_model(args, model_name: str, train_ds, val_ds, device: torch.device):
    print(f"\n=== Training model: {model_name} ===")
    num_classes = len(train_ds.label_map) if hasattr(train_ds, "label_map") else len(train_ds.dataset.label_map)
    model = None

    if args.dann:    
        model = RoadworkClassifier(
            backbone_name=model_name,
            pretrained=args.pretrained,
            num_classes=num_classes,
            head_hidden=args.head_hidden,
            domain_adapt=True,
            domain_hidden=args.domain_hidden,
            grl_lambda=0.0,
            dropout=args.head_dropout,
        )
    else:
        # use simple backbone builder
        model_builder = get_backbone_builder(model_name)
        try:
            model = model_builder(num_classes=num_classes, pretrained=args.pretrained)
        except Exception as e:
            print(f"Warning: failed to build {model_name} pretrained; fallback to pretrained=False ({e})")
            model = model_builder(num_classes=num_classes, pretrained=False)

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

    scaler = GradScaler("cuda", enabled=args.mixed_precision)

    # DataLoaders: Subset objects may be passed (your main script uses Subset)
    train_sampler = BalancedDomainSampler(train_ds.real_indices, train_ds.synthetic_indices, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_sampler = BalancedDomainSampler(val_ds.real_indices, val_ds.synthetic_indices, batch_size=args.batch_size, shuffle=False, drop_last=False)
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

    # ---------- If DANN: pretrain stage ----------
    global_step = 0
    total_pretrain_steps = len(train_loader) * max(1, args.pretrain_epochs) if args.dann else 0

    no_progress = 0
    if args.dann:
        print("Starting DANN pretraining for", args.pretrain_epochs, "epochs")
        for epoch in range(args.pretrain_epochs):
            model.train()
            dann_metrics, global_step = train_epoch_dann(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                global_step=global_step,
                total_steps=total_pretrain_steps,
                lambda_domain_weight=args.lambda_domain,
                tb_writer=writer,
                weights=weights_tensor,
            )
            val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)
            print(f"[DANN pretrain {epoch+1}/{args.pretrain_epochs}] train_f1={dann_metrics.get('f1',0):.4f} train_acc={dann_metrics.get('accuracy',0):.4f} train_loss={dann_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f}")
            # TB log
            writer.add_scalar("pretrain/train_f1", dann_metrics.get("f1", 0), epoch)
            writer.add_scalar("pretrain/train_acc", dann_metrics.get("accuracy", 0), epoch)
            writer.add_scalar("pretrain/train_prec", dann_metrics.get("precision", 0), epoch)
            writer.add_scalar("pretrain/train_rec", dann_metrics.get("recall", 0), epoch)
            writer.add_scalar("pretrain/train_auc", dann_metrics.get("auc", 0), epoch)
            writer.add_scalar("pretrain/train_loss", dann_metrics["loss"], epoch)
            
            writer.add_scalar("pretrain/val_f1", val_metrics["f1"], epoch)
            writer.add_scalar("pretrain/val_prec", val_metrics["precision"], epoch)
            writer.add_scalar("pretrain/val_rec", val_metrics["recall"], epoch)
            writer.add_scalar("pretrain/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar("pretrain/val_acc", val_metrics["accuracy"], epoch)
            writer.add_scalar("pretrain/val_mcc", val_metrics["mcc"], epoch)
            # save last
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, os.path.join(args.output, model_name + "_last.pth"))
            # save best by val f1 (or val acc depending on arg)
            cur_mcc = val_metrics.get("mcc", 0)
            cur_acc = val_metrics.get("accuracy", 0)
            cur = (cur_mcc + cur_acc) / 2.0
            if cur > best_val_acc:
                best_val_acc = cur
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, best_ckpt_path)
                print("Saved best pretrain checkpoint ->", best_ckpt_path)
                no_progress = 0
            else:
                no_progress += 1

            if no_progress >= 2:
                print("No progress in validation for 5 epochs, stopping DANN pretraining early.")
                break

    # ---------- Supervised training (or if not DANN: full supervised) ----------
    print("Starting supervised training for", args.epochs, "epochs")
    best_val_acc = 0.0
    no_progress = 0
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

            # Augment: Mixup / CutMix
            do_mix = random.random() < args.mixup_prob
            do_cut = (not do_mix) and (random.random() < args.cutmix_prob)
            if do_mix:
                imgs, y_a, y_b, _, lam = mixup_data(imgs, labels, alpha=args.mixup_alpha)
            elif do_cut:
                imgs, y_a, y_b, _, lam = cutmix_data(imgs, labels, alpha=args.cutmix_alpha)
            else:
                y_a = y_b = None
                lam = 1.0

            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=args.mixed_precision):
                out = model(imgs)
                # model returns either logits or (logits, ...) in case DANN
                if isinstance(out, tuple):
                    class_logits = out[0]
                else:
                    class_logits = out
                if y_a is not None and y_b is not None:
                    loss = mixup_criterion(criterion, class_logits, y_a.to(device), y_b.to(device), lam)
                else:
                    loss = criterion(class_logits, labels)

            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--pretrain-epochs", type=int, default=8, help="If >0 and --dann, number of DANN pretrain epochs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0)
    parser.add_argument("--mixup-prob", type=float, default=0)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--tb-logdir", type=str, default=None, help="TensorBoard root dir")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Enable resume from last checkpoint if exists")
    parser.add_argument("--dann", action="store_true", help="Enable domain-adversarial pretraining (requires model.roadwalk.RoadworkClassifier)")
    parser.add_argument("--lambda-domain", type=float, default=1.0, help="Domain loss weight during DANN pretrain")
    parser.add_argument("--synthetic-indicators", nargs="*", default=["synthetic", "synth", "ai_", "t2i", "i2i", "generated"])
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.4)
    parser.add_argument("--domain-hidden", type=int, default=256)
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP")
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=0.0)
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
