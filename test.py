# test.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datasets.natix_dataset import NatixDataset
from model.backbone import get_backbone_builder
from model.utils import collate_fn, evaluate_ensemble, filter_real_only_indices, make_transforms, denormalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoints", nargs="+", required=True, help="Paths to trained model checkpoints for ensemble")
    parser.add_argument("--model-name", type=str, required=True, help="Backbone architecture name used for training")
    parser.add_argument("--data-dirs", nargs="+", required=True, help="Directories containing test images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tb-logdir", type=str, default=None)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--is-synthetic-key", type=str, default="is_synthetic")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    transform = make_transforms(args.image_size)
    test_ds = NatixDataset(dirs=args.data_dirs, transform=transform)

    if args.real_only:
        idxs = filter_real_only_indices(test_ds, is_synthetic_key=args.is_synthetic_key)
        test_ds = Subset(test_ds, idxs)
        print(f"Real-only test dataset: {len(test_ds)} samples")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    # -------------------------
    # Ensemble evaluation
    # -------------------------
    model_builder = get_backbone_builder(args.model_name)
    num_classes = len(test_ds.dataset.label_map) if isinstance(test_ds, Subset) else len(test_ds.label_map)

    metrics = evaluate_ensemble(args.model_checkpoints, model_builder, test_loader, device, num_classes)
    print("\n===== Ensemble Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # -------------------------
    # TensorBoard logging
    # -------------------------
    if args.tb_logdir:
        os.makedirs(args.tb_logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, "ensemble_test"))
        for k,v in metrics.items():
            if isinstance(v,float):
                writer.add_scalar(f"test/{k}", v, 0)
        writer.close()
        print(f"Test metrics logged to {args.tb_logdir}")
