# train.py
import os
import argparse
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
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
    )

# -------------------------
# Single model training
# -------------------------
def train_single_model(args, model_name:str, train_ds: NatixDataset, val_ds: NatixDataset, device: torch.device):
    print(f"\n=== Training model: {model_name} ===")
    model_builder = get_backbone_builder(model_name)
    num_classes = len(train_ds.label_map)
    model = model_builder(num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    writer = None
    if args.tb_logdir:
        os.makedirs(args.tb_logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, model_name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")))

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, model_name+"_best.pth")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            images, labels, metas, paths = batch
            images, labels = images.to(device), torch.tensor([(-1 if l is None else l) for l in labels],dtype=torch.long).to(device)
            mask = labels>=0
            if mask.sum()==0: continue
            images, labels = images[mask], labels[mask]

            # Mixup / CutMix
            if random.random()<args.mixup_prob:
                images, y_a, y_b, _, lam = mixup_data(images, labels, alpha=args.mixup_alpha)
            elif random.random()<args.cutmix_prob:
                images, y_a, y_b, _, lam = cutmix_data(images, labels, alpha=args.cutmix_alpha)
            else:
                y_a, y_b, lam = None, None, 1.0

            optimizer.zero_grad()
            outputs = model(images)
            if y_a is not None and y_b is not None:
                loss = mixup_criterion(criterion, outputs, y_a.to(device), y_b.to(device), lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if writer and step%args.log_every_steps==0:
                writer.add_scalar("train/loss", loss.item(), epoch*len(train_loader)+step)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, num_classes)
        print(f"[{epoch+1}/{args.epochs}] Val Acc: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")
        if writer:
            for k,v in val_metrics.items():
                if isinstance(v,float):
                    writer.add_scalar(f"val/{k}", v, epoch)
            # log sample images
            if val_metrics['samples']:
                img_tensor = torch.stack([s[0] for s in val_metrics['samples']])
                img_tensor = denormalize(img_tensor)
                writer.add_images("val/samples", img_tensor, epoch)
            if val_metrics['misclassified']:
                img_tensor = torch.stack([s[0] for s in val_metrics['misclassified']])
                img_tensor = denormalize(img_tensor)
                writer.add_images("val/misclassified", img_tensor, epoch)

        if val_metrics["accuracy"]>best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"epoch":epoch}, best_ckpt_path)

    if writer: writer.close()
    print(f"Model {model_name} best checkpoint saved at: {best_ckpt_path}")
    return best_ckpt_path

# -------------------------
# Main
# -------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="List of backbone model names to train and ensemble")
    parser.add_argument("--train-dirs", nargs="+", required=True, help="Directories for training images")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data used for validation (0–1)")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--tb-logdir", type=str, default=None)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--log-image-count", type=int, default=16)
    parser.add_argument("--real-only-val", action="store_true")
    parser.add_argument("--is-synthetic-key", type=str, default="is_synthetic")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # -------------------------
    # Build datasets
    # -------------------------
    transform_train = make_transforms(args.image_size)
    transform_val = make_transforms(args.image_size)

    full_ds = NatixDataset(dirs=args.train_dirs, transform=transform_train, augment=args.augment)

    num_samples = len(full_ds)
    indices = list(range(num_samples))
    random.shuffle(indices)

    split = int(num_samples * (1 - args.val_split))
    train_indices, val_indices = indices[:split], indices[split:]

    # Create subsets
    train_ds = Subset(full_ds, train_indices)

    # For validation, we use the same dataset but with val transform (no augmentation)
    val_base = NatixDataset(dirs=args.train_dirs, transform=transform_val, label_map=full_ds.label_map)
    val_ds = Subset(val_base, val_indices)

    if args.real_only_val:
        idxs = filter_real_only_indices(val_ds, is_synthetic_key=args.is_synthetic_key)
        val_ds = Subset(val_ds, idxs)
        print(f"Real-only validation: {len(val_ds)} samples")

    # -------------------------
    # Train multiple models
    # -------------------------
    best_ckpts = []
    for model_name in args.models:
        best_ckpt = train_single_model(args, model_name, train_ds, val_ds, device)
        best_ckpts.append(best_ckpt)

    # -------------------------
    # Evaluate ensemble
    # -------------------------
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    ensemble_metrics = evaluate_ensemble(best_ckpts, get_backbone_builder(args.models[0]), val_loader, device, num_classes=len(train_ds.label_map))
    print("\n===== Ensemble Metrics =====")
    print(ensemble_metrics)
