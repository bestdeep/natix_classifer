# test.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datasets.natix_dataset import NatixDataset
from model.backbone import get_backbone_builder
from model.utils import collate_fn, make_transforms, evaluate, safe_model_load
from model.roadwalk import RoadworkClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Backbone architecture name used for training")
    parser.add_argument("--test-path", required=True, type=str, help="File list path containing test images")
    parser.add_argument("--checkpoints", required=True, type=str, help="Trained weights for backbone")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tb-logdir", type=str, default=None)
    parser.add_argument("--dann", action="store_true", help="Enable domain-adversarial pretraining (requires model.roadwalk.RoadworkClassifier)")
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.4)
    parser.add_argument("--domain-hidden", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    transform = make_transforms(args.image_size)
    test_ds = NatixDataset(dirs=args.test_path, transform=transform, augment=True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    num_classes = len(test_ds.label_map)
    if args.dann:    
        model = RoadworkClassifier(
            backbone_name=args.model,
            num_classes=num_classes,
            head_hidden=args.head_hidden,
            domain_adapt=True,
            domain_hidden=args.domain_hidden,
            grl_lambda=0.0,
            dropout=args.head_dropout,
        )
    else:
        # use simple backbone builder
        model_builder = get_backbone_builder(args.model)
        try:
            model = model_builder(num_classes=num_classes, pretrained=args.pretrained)
        except Exception as e:
            print(f"Warning: failed to build {args.model} pretrained; fallback to pretrained=False ({e})")
            model = model_builder(num_classes=num_classes, pretrained=False)

    model = model.to(device)
    # load weights
    checkpoint = torch.load(args.checkpoints, map_location=device)
    model = safe_model_load(model, checkpoint["model_state"])
    model.eval()

    metrics = evaluate(model, test_loader, device, num_classes, log_image_count=0)
    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # -------------------------
    # TensorBoard logging
    # -------------------------
    if args.tb_logdir:
        os.makedirs(args.tb_logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, "test"))
        for k,v in metrics.items():
            if isinstance(v,float):
                writer.add_scalar(f"test/{k}", v, 0)
        writer.close()
        print(f"Test metrics logged to {args.tb_logdir}")
