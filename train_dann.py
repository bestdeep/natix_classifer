# train_combined.py
import os
import argparse
import random
from datetime import datetime
from typing import List, Optional
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T

# user project imports (assumed present)
from model.backbone import get_backbone_builder
from datasets.natix_dataset import NatixDataset
from model.utils import (
    set_seed,
    make_transforms,
    denormalize,
    mixup_data,
    cutmix_data,
    mixup_criterion,
    collate_fn,
    filter_real_only_indices,
    evaluate,
    evaluate_ensemble,
    normalize_arg_list,
    FocalLoss,
)

# Optional DANN model (only imported if --dann)
# from models.custom_model import RoadworkClassifier  <-- imported lazily when needed

# -------------------------
# Helpers
# -------------------------
def infer_domain_from_meta_or_path(meta: dict, path: str, synthetic_indicators: Optional[List[str]] = None) -> int:
    """
    Heuristic: 0 -> real, 1 -> synthetic
    Checks meta keys (is_synthetic, domain) then path substrings
    """
    if meta is None:
        meta = {}
    v = meta.get("is_synthetic")
    if isinstance(v, bool):
        return 1 if v else 0
    dom = meta.get("domain")
    if isinstance(dom, str):
        d = dom.lower()
        if "synth" in d or "synthetic" in d:
            return 1
        if "real" in d:
            return 0
    p = (path or "").lower()
    if synthetic_indicators:
        for s in synthetic_indicators:
            if s.lower() in p:
                return 1
    for s in ("synthetic", "synth", "ai_", "t2i", "i2i", "generated", "fake"):
        if s in p:
            return 1
    return 0

def ensure_tensor_labels(labels):
    """Return torch.LongTensor labels; -1 for missing"""
    if isinstance(labels, torch.Tensor):
        if labels.dtype != torch.long:
            return labels.long()
        return labels
    # assume list-like
    return torch.tensor([(-1 if l is None else int(l)) for l in labels], dtype=torch.long)

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
    synthetic_indicators: List[str],
    tb_writer: Optional[SummaryWriter],
    epoch_idx: int,
):
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    running_loss = 0.0
    y_true, y_pred, y_scores = [], [], []
    pbar = tqdm(loader, desc=f"DANN-train", leave=False)
    for step, batch in enumerate(pbar):
        imgs, labels, metas, paths = batch
        imgs = imgs.to(device)
        labels = ensure_tensor_labels(labels).to(device)
        mask = labels >= 0

        # domain labels
        domain_labels = [infer_domain_from_meta_or_path(m, p, synthetic_indicators) for m, p in zip(metas, paths)]
        domain_labels = torch.tensor(domain_labels, dtype=torch.long, device=device)

        # dynamic GRL lambda schedule (DANN paper common schedule)
        p = min(1.0, float(global_step) / float(max(1, total_steps)))
        grl_lambda = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        # set grl if model supports it
        if hasattr(model, "set_grl_lambda"):
            model.set_grl_lambda(float(grl_lambda))

        # Mixup/CutMix still supported for DANN if desired (optional)
        # Here we keep simple: no MixUp/CutMix in DANN stage to avoid domain label confusion.
        with autocast():
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
            # use detach().item() to avoid autograd-to-scalar warnings
            tb_writer.add_scalar("train/dann_cls_loss", float(cls_loss.detach().item()) if isinstance(cls_loss, torch.Tensor) else float(cls_loss), global_step)
            tb_writer.add_scalar("train/dann_domain_loss", float(domain_loss.detach().item()) if isinstance(domain_loss, torch.Tensor) else float(domain_loss), global_step)
            tb_writer.add_scalar("train/dann_loss", float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss), global_step)

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        if mask.sum() > 0:
            probs = torch.softmax(class_logits[mask], dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(class_logits[mask], dim=1).detach().cpu().numpy().tolist()
            trues = labels[mask].cpu().numpy().tolist()
            y_scores.extend(probs.tolist())
            y_pred.extend(preds)
            y_true.extend(trues)

        global_step += 1
        pbar.set_postfix(loss=running_loss / (step + 1))
    metrics = {}
    if len(y_true) > 0:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        auc = None
        try:
            if len(set(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = None
        metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "auc": auc}
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics, global_step

# -------------------------
# Single model training (supervised or DANN)
# -------------------------
def train_single_model(args, model_name: str, train_ds, val_ds, device: torch.device):
    print(f"\n=== Training model: {model_name} ===")
    num_classes = len(train_ds.label_map) if hasattr(train_ds, "label_map") else len(train_ds.dataset.label_map)
    model = None

    if args.dann:
        # use RoadworkClassifier wrapping the chosen backbone (lazy import)
        from model.roadwalk import RoadworkClassifier
        model = RoadworkClassifier(
            backbone_name=args.backbone if args.backbone else model_name,
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
    criterion = FocalLoss(weight=None, gamma=2.0, reduction="mean")
    scaler = GradScaler(enabled=args.mixed_precision)

    # DataLoaders: Subset objects may be passed (your main script uses Subset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # TensorBoard
    writer = None
    if args.tb_logdir:
        os.makedirs(args.tb_logdir, exist_ok=True)
        tb_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, tb_name))
        print("TensorBoard logs ->", os.path.join(args.tb_logdir, tb_name))

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, model_name + "_best.pth")
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume if exists
    if args.resume and os.path.isfile(best_ckpt_path):
        print(f"Resuming training from checkpoint: {best_ckpt_path}")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state", {}))

    # ---------- If DANN: pretrain stage ----------
    global_step = 0
    total_pretrain_steps = len(train_loader) * max(1, args.pretrain_epochs) if args.dann else 0

    if args.dann:
        print("Starting DANN pretraining for", args.pretrain_epochs, "epochs")
        for epoch in range(args.pretrain_epochs):
            dann_metrics, global_step = train_epoch_dann(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                global_step=global_step,
                total_steps=total_pretrain_steps,
                lambda_domain_weight=args.lambda_domain,
                synthetic_indicators=args.synthetic_indicators,
                tb_writer=writer,
                epoch_idx=epoch,
            )
            val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)
            print(f"[DANN pretrain {epoch+1}/{args.pretrain_epochs}] train_f1={dann_metrics.get('f1',0):.4f} train_loss={dann_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f}")
            # TB log
            if writer:
                writer.add_scalar("pretrain/train_loss", dann_metrics["loss"], epoch)
                writer.add_scalar("pretrain/val_loss", val_metrics["loss"], epoch)
                writer.add_scalar("pretrain/val_acc", val_metrics["accuracy"], epoch)
            # save last
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, os.path.join(args.output_dir, model_name + "_last.pth"))
            # save best by val f1 (or val acc depending on arg)
            save_metric = val_metrics.get(args.save_metric, val_metrics["accuracy"])
            if args.save_metric == "f1":
                # prefer f1 if available
                cur = val_metrics.get("f1", val_metrics["accuracy"])
            else:
                cur = save_metric
            if cur > best_val_acc:
                best_val_acc = cur
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, best_ckpt_path)
                print("Saved best pretrain checkpoint ->", best_ckpt_path)

    # ---------- Supervised training (or if not DANN: full supervised) ----------
    print("Starting supervised training for", args.epochs, "epochs")
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        running_loss, correct, total = 0.0, 0, 0
        for step, batch in enumerate(loop):
            imgs, labels, metas, paths = batch
            imgs = imgs.to(device)
            labels = ensure_tensor_labels(labels).to(device)
            mask = labels >= 0
            if mask.sum() == 0:
                continue
            imgs, labels = imgs[mask], labels[mask]

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
            with autocast(enabled=args.mixed_precision):
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
                global_step += 1
                writer.add_scalar("train/step_loss", float(loss.item()), global_step)

            acc = 100.0 * correct / total if total > 0 else 0.0
            loop.set_postfix(loss=float(loss.item()), acc=acc)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        scheduler.step()

        # validation
        val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")

        # tensorboard logging
        if writer:
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("train/epoch_acc", train_acc, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    writer.add_scalar(f"val/{k}", v, epoch)
            # log sample images (normalized -> denormalize)
            if val_metrics.get("samples"):
                imgs_to_log = torch.stack([s[0] for s in val_metrics["samples"]])
                imgs_to_log = denormalize(imgs_to_log)
                writer.add_images("val/samples", imgs_to_log, epoch)
            if val_metrics.get("misclassified"):
                imgs_mis = torch.stack([s[0] for s in val_metrics["misclassified"]])
                imgs_mis = denormalize(imgs_mis)
                writer.add_images("val/misclassified", imgs_mis, epoch)
                writer.add_text("val/misclassified_pred", "\n".join([str(s[1]) for s in val_metrics["misclassified"]]), epoch)
                writer.add_text("val/misclassified_true", "\n".join([str(s[2]) for s in val_metrics["misclassified"]]), epoch)
                writer.add_text("val/misclassified_path", "\n".join([str(s[3]) for s in val_metrics["misclassified"]]), epoch)

        # checkpointing
        # save last
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, os.path.join(args.output_dir, model_name + "_last.pth"))

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, best_ckpt_path)
            print("Saved new best checkpoint ->", best_ckpt_path)

    if writer:
        writer.close()
    print(f"Model {model_name} best checkpoint saved at: {best_ckpt_path}")
    return best_ckpt_path

# -------------------------
# Main entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="List of backbone model names to train and ensemble")
    parser.add_argument("--train-dirs", nargs="+", required=True, help="Directories for training images")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data used for validation (0–1)")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--pretrain-epochs", type=int, default=0, help="If >0 and --dann, number of DANN pretrain epochs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--tb-logdir", type=str, default=None, help="TensorBoard root dir")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--log-image-count", type=int, default=16)
    parser.add_argument("--real-only-val", action="store_true")
    parser.add_argument("--is-synthetic-key", type=str, default="is_synthetic")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dann", action="store_true", help="Enable domain-adversarial pretraining (requires models.custom_model.RoadworkClassifier)")
    parser.add_argument("--lambda-domain", type=float, default=1.0, help="Domain loss weight during DANN pretrain")
    parser.add_argument("--synthetic-indicators", nargs="*", default=["synthetic", "synth", "ai_", "t2i", "i2i", "generated"])
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.4)
    parser.add_argument("--domain-hidden", type=int, default=256)
    parser.add_argument("--save-metric", dest="save_metric", choices=["f1", "accuracy", "loss"], default="f1")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP")
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=0.0)
    parser.add_argument("--pretrained-backbone", dest="pretrained", action="store_true")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone name used for DANN (if --dann), otherwise each model in --models is used")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    args.train_dirs = normalize_arg_list(args.train_dirs)
    args.models = normalize_arg_list(args.models)
    print("Training directories:", args.train_dirs)
    print("Models to train:", args.models)

    # transforms
    transform_train = make_transforms(args.image_size)
    transform_val = make_transforms(args.image_size)

    # build dataset
    full_ds = NatixDataset(dirs=args.train_dirs, transform=transform_train, augment=args.augment)
    num_samples = len(full_ds)
    indices = list(range(num_samples))
    random.shuffle(indices)

    split = int(num_samples * (1 - args.val_split))
    train_indices, val_indices = indices[:split], indices[split:]

    train_ds = Subset(full_ds, train_indices)
    val_base = NatixDataset(dirs=args.train_dirs, transform=transform_val, label_map=full_ds.label_map)
    val_ds = Subset(val_base, val_indices)

    # attach label_map to subsets for compatibility
    train_ds.label_map = full_ds.label_map
    val_ds.label_map = full_ds.label_map

    print(f"Total samples: {num_samples}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    if args.real_only_val:
        idxs = filter_real_only_indices(val_ds, is_synthetic_key=args.is_synthetic_key)
        val_ds = Subset(val_ds, idxs)
        print(f"Real-only validation: {len(val_ds)} samples")

    best_ckpts = []
    for model_name in args.models:
        # If using DANN and backbone provided, pass backbone name to DANN builder; otherwise use model_name
        model_builder_name = args.backbone if args.dann and args.backbone else model_name
        ckpt = train_single_model(args, model_builder_name, train_ds, val_ds, device)
        best_ckpts.append(ckpt)

    # Evaluate ensemble using first model builder for architecture (evaluate_ensemble expects a builder)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    # for evaluate_ensemble we pass the get_backbone_builder of the first model in args.models (fallback to args.backbone)
    backbone_for_ensemble = args.backbone if args.backbone else args.models[0]
    builder = get_backbone_builder(backbone_for_ensemble)
    ensemble_metrics = evaluate_ensemble(best_ckpts, builder, val_loader, device, num_classes=len(train_ds.label_map))
    print("\n===== Ensemble Metrics =====")
    print(ensemble_metrics)
