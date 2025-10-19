import torch
import torchvision.transforms as T
import torch.nn.functional as F
from typing import List, Union
from time import time
import numpy as np

from model.roadwalk import RoadworkClassifier

class NatixClassifier:
    def __init__(self, backbone_name: str, device: str = "cpu", debug: bool = False):
        """
        Args:
            backbone_name: used to instantiate RoadworkClassifier and to find checkpoint file.
            device: "cpu" or "cuda"
            debug: if True prints shapes/values for debugging
        """
        self.device = torch.device(device)
        self.debug = debug

        # instantiate model (user must have RoadworkClassifier in scope)
        self.model = RoadworkClassifier(
            backbone_name=backbone_name,
            pretrained=False,
            domain_adapt=True,
        )

        model_path = f"checkpoints/{backbone_name}_best.pth"
        # load checkpoint robustly
        checkpoint = torch.load(model_path, map_location=self.device)
        # support multiple common keys
        state_keys = ['model_state', 'state_dict', 'model']
        state = None
        for k in state_keys:
            if k in checkpoint:
                state = checkpoint[k]
                break
        if state is None:
            # maybe checkpoint itself is the state_dict
            state = checkpoint

        # sometimes saved with 'module.' prefixes (from DataParallel) — handle that
        try:
            self.model.load_state_dict(state)
        except RuntimeError as e:
            # try stripping 'module.' prefix if present
            new_state = {}
            for k, v in state.items():
                new_key = k
                if k.startswith("module."):
                    new_key = k[len("module."):]
                new_state[new_key] = v
            self.model.load_state_dict(new_state)

        self.model.to(self.device)
        self.model.eval()

        # normalization transform (expects input in [0,1])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image(s) to a normalized torch tensor on CPU.
        Accepts:
            image: (H,W,C) or (B,H,W,C). C can be 1 or 3. dtype can be uint8 or float.
        Returns:
            tensor: torch.FloatTensor shape (B, C, H, W) with values normalized.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")

        # unify to batch dimension
        if image.ndim == 3:
            # (H,W,C) -> (1,H,W,C)
            img = image[np.newaxis, ...]
        elif image.ndim == 4:
            img = image
        else:
            raise ValueError("Input image must have 3 (H,W,C) or 4 (B,H,W,C) dimensions.")

        # ensure channel last
        if img.shape[-1] not in (1, 3):
            raise ValueError(f"Expected last dim channels=1 or 3, got {img.shape[-1]}")

        # convert to float32 and scale to [0,1] if needed
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        # if grayscale (1 channel), repeat to 3 channels
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)  # (B,H,W,3)

        # to tensor: (B, C, H, W)
        tensor = torch.from_numpy(img).permute(0, 3, 1, 2).float()  # (B,C,H,W)

        # apply per-channel normalization using torchvision Normalize which expects tensor in [0,1]
        # Normalize works on a single tensor; apply batch-wise by iterating or using functional
        # We'll do batch-wise without loops: Normalize supports batched tensors.
        tensor = self.normalize(tensor)
        return tensor

    def _predict_batch(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Run model forward pass on a batch tensor and return numpy array of probabilities for class 1.
        input_tensor expected shape: (B, C, H, W)
        Returns: np.ndarray shape (B,) of probabilities for class index 1.
        """
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output, _ = self.model(input_tensor)  # expected shape (B, num_classes)
            # If model already returns logits, apply softmax to get probabilities
            probs = F.softmax(output, dim=1)  # (B, num_classes)
            probs_cpu = probs.cpu().numpy()

        # By convention return probability of class 1. If your positive-class index is different, adjust idx. 1: Roadwork, 0: No Roadwork
        pos_idx = 1
        if probs_cpu.shape[1] <= pos_idx:
            # if only one output (sigmoid style), treat second class absent -> take single-column as prob
            if probs_cpu.shape[1] == 1:
                # if model returned single logit per sample, apply sigmoid
                single_logits = output.cpu().numpy().flatten()
                probs_pos = 1.0 / (1.0 + np.exp(-single_logits))
                return probs_pos
            else:
                raise ValueError(f"Model output has shape {probs_cpu.shape}; cannot extract class {pos_idx}")
        return probs_cpu[:, pos_idx]

    def predict(self, image: np.ndarray) -> Union[float, np.ndarray]:
        """
        Args:
            image: numpy array (H,W,C) or (B,H,W,C). Channels 1 or 3. Values either 0-255 uint8 or floats in 0-1.
        Returns:
            If single image input: float probability of roadwork (class 1).
            If batch input: np.ndarray of probabilities (shape B,).
        """
        input_tensor = self._preprocess(image)  # (B,C,H,W)
        if self.debug:
            print(f"[NatixClassifier] input tensor shape: {input_tensor.shape}")

        probs = self._predict_batch(input_tensor)  # np.ndarray shape (B,)
        # if original input was a single image, return scalar
        if image.ndim == 3:
            return float(probs[0])
        else:
            return probs

def test_one(backbone_name: str = "swin_large_patch4_window7_224", image_path: str = None):
    from PIL import Image
    # Example usage
    start_time = time()
    classifier = NatixClassifier(backbone_name=backbone_name, device="cpu")
    end_time = time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")

    # Dummy input tensor
    image = Image.open(image_path).convert("RGB")
    # image = image.resize((224, 224))
    image_np = np.array(image)
    # dummy_input = np.expand_dims(image_np, axis=0)  # (1,H,W,C)
    start_time = time()
    prediction = classifier.predict(image_np)
    end_time = time()
    print(f"Prediction made in {end_time - start_time:.2f} seconds.")
    print(f"Prediction: {prediction}")

def test_dir(dir_paths: List[str], backbone_name: str = "swin_large_patch4_window7_224", debug: bool = False):
    import os
    from datasets.natix_dataset import NatixDataset
    from model.utils import make_transforms, collate_fn
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    from tqdm import tqdm

    # Batch prediction on all images in a directory
    start_time = time()
    classifier = NatixClassifier(backbone_name=backbone_name, device="cpu", debug=True)
    end_time = time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")
    
    transform = make_transforms(224)
    test_ds = NatixDataset(dirs=dir_paths, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)

    start_time = time()
    all_preds, all_trues, all_probs = [], [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="val", leave=False)
        for images, labels, metas, paths in pbar:
            images = images.to("cpu")

            # Convert labels: -1 for unknown
            labels_tensor = torch.tensor([(-1 if l is None else l) for l in labels], dtype=torch.long)
            mask = labels_tensor >= 0
            if mask.sum() == 0:
                continue
            labels_tensor = labels_tensor.to("cpu")

            # Predict probabilities for class 1
            pred = classifier._predict_batch(images)
            neg_pred = 1 - pred

            # Stack into [[class_0_prob, class_1_prob], ...]
            probs = np.column_stack((neg_pred, pred))

            if debug:
                print(f"labels: {labels}")
                print(f"pred: {pred}")

            # Convert probabilities to discrete labels for metrics
            pred_labels = (pred >= 0.5).astype(int)

            # Collect results
            all_preds.extend(pred_labels.tolist())
            all_trues.extend(labels_tensor.cpu().tolist())
            all_probs.extend(probs.tolist())

    # Compute metrics
    if all_preds:
        acc = accuracy_score(all_trues, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_trues, all_preds, average="binary", zero_division=0
        )

        all_probs_array = np.array(all_probs)
        roc_auc = roc_auc_score(all_trues, all_probs_array[:, 1])
    else:
        acc = precision = recall = f1 = roc_auc = 0.0

    end_time = time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    result = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    }
    print("Test results:", result)
    return result

if __name__ == "__main__":
    import argparse
    from model.utils import normalize_arg_list
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    test_one_parser = subparsers.add_parser("test_one", help="Test single image")
    test_one_parser.add_argument("--backbone", type=str, default="swin_large_patch4_window7_224",
                        help="Backbone name for the RoadworkClassifier model.")
    test_one_parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the image file to test.")

    test_dir_parser = subparsers.add_parser("test_dir", help="Test all images in a directory")
    test_dir_parser.add_argument("--backbone", type=str, default="swin_large_patch4_window7_224",
                        help="Backbone name for the RoadworkClassifier model.")
    test_dir_parser.add_argument("--image-dirs", nargs="+", type=str, required=True,
                        help="Directories for test images")

    args = parser.parse_args()

    if args.command == "test_one":
        test_one(backbone_name=args.backbone, image_path=args.image_path)
    elif args.command == "test_dir":
        image_dirs = normalize_arg_list(args.image_dirs)
        test_dir(dir_paths=image_dirs, backbone_name=args.backbone)