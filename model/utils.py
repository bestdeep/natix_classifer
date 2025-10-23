import numpy as np
import random
from typing import List, Any, Optional
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from PIL import Image
from torch.utils.data import DataLoader
import re
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_transforms(image_size: int = 224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    def transform(x):
        # tensor input
        if isinstance(x, torch.Tensor):
            if x.shape[1] != image_size or x.shape[2] != image_size:
                x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(image_size,image_size), mode="bilinear", align_corners=False).squeeze(0)
            return normalize(x)
        # numpy input
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                x = (x*255).astype(np.uint8) if x.max() <= 1.01 else x.astype(np.uint8)
            x = T.ToTensor()(Image.fromarray(x))
            if x.shape[1] != image_size or x.shape[2] != image_size:
                x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(image_size,image_size), mode="bilinear", align_corners=False).squeeze(0)
            return normalize(x)
        # PIL input
        if isinstance(x, Image.Image):
            x = T.ToTensor()(x)
            if x.shape[1] != image_size or x.shape[2] != image_size:
                x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(image_size,image_size), mode="bilinear", align_corners=False).squeeze(0)
            return normalize(x)
        # fallback
        try:
            x = T.ToTensor()(Image.fromarray(np.asarray(x)))
            if x.shape[1] != image_size or x.shape[2] != image_size:
                x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(image_size,image_size), mode="bilinear", align_corners=False).squeeze(0)
            return normalize(x)
        except Exception as e:
            raise TypeError(f"Unsupported image type for transform: {type(x)} - {e}")
    return transform

def denormalize(tensor: torch.Tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    if tensor.ndim == 3: tensor = tensor.unsqueeze(0); squeezed=True
    else: squeezed=False
    mean_t = torch.tensor(mean, device=tensor.device).view(1,-1,1,1)
    std_t = torch.tensor(std, device=tensor.device).view(1,-1,1,1)
    img = tensor*std_t + mean_t
    img = torch.clamp(img, 0.0, 1.0)
    return img.squeeze(0) if squeezed else img

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float=0.2):
    if alpha <= 0: return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1-lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index, lam

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float=1.0):
    if alpha <= 0: return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, W, H = x.size()
    index = torch.randperm(batch_size).to(x.device)
    cut_rat = np.sqrt(1.0-lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx-cut_w//2,0,W), np.clip(cy-cut_h//2,0,H)
    x2, y2 = np.clip(cx+cut_w//2,0,W), np.clip(cy+cut_h//2,0,H)
    x[:,:,x1:x2,y1:y2] = x[index,:,x1:x2,y1:y2]
    lam_adjusted = 1.0 - ((x2-x1)*(y2-y1)/(W*H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, index, lam_adjusted

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred,y_a)+(1-lam)*criterion(pred,y_b)

def collate_fn(batch):
    """
    Batch item: (img_t, label_index, meta_dict, image_path)
    We'll output:
       images: tensor (B,C,H,W)
       labels: torch.LongTensor (B,) where -1 indicates missing label
       paths: list of strings
    """
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    labels_t = torch.tensor(labels, dtype=torch.long)
    return imgs, labels_t, paths

def collate_fn_multi(batch):
    """
    Batch is list of tuples: (images_k, label_idx, meta, path, domain)
     - images_k: (K, C, H, W)
    Returns:
      images: Tensor (B, K, C, H, W)
      labels: LongTensor (B,)
      paths: list[str]
      domains: LongTensor (B,)  (0 real, 1 synthetic)
    """
    images_k = [item[0] for item in batch]  # each is (K,C,H,W)
    labels = [(-1 if item[1] is None else int(item[1])) for item in batch]
    paths = [item[2] for item in batch]
    domains = [int(item[3]) for item in batch]

    images_stack = torch.stack(images_k, dim=0)  # shape (B, K, C, H, W)
    labels_t = torch.tensor(labels, dtype=torch.long)
    domains_t = torch.tensor(domains, dtype=torch.long)

    return images_stack, labels_t, paths, domains_t

# -------------------------
# Evaluation
# -------------------------
def safe_model_load(model, checkpoint):
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
        model.load_state_dict(state)
    except RuntimeError as e:
        # try stripping 'module.' prefix if present
        new_state = {}
        for k, v in state.items():
            new_key = k
            if k.startswith("module."):
                new_key = k[len("module."):]
            new_state[new_key] = v
        model.load_state_dict(new_state)

    return model


def evaluate(model, dataloader, device, num_classes):
    """
    Evaluate model on dataloader where each batch yields:
      images_stack: (B, K, C, H, W)
      labels: (B,)  (ints or None)
      paths: list
      domains: optional

    This function averages logits across K augmentations per sample and computes metrics per-sample.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    all_preds = []
    all_probs = []
    all_trues = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="val", leave=False)
        for images_stack, labels, paths, _ in pbar:
            # move to device
            B, K, C, H, W = images_stack.shape
            imgs_flat = images_stack.view(B * K, C, H, W).to(device)   # (B*K, C, H, W)
            labels = ensure_tensor_labels(labels)                       # (B,)
            # Create per-sample mask (True = labeled)
            mask_per_sample = (labels >= 0)                             # (B,)

            # forward in one big batch
            out = model(imgs_flat)
            if isinstance(out, (tuple, list)):
                class_logits_flat = out[0]
            else:
                class_logits_flat = out

            # shape checks
            if class_logits_flat.ndim != 2 or class_logits_flat.size(0) != B * K:
                raise RuntimeError(f"Expected class_logits_flat shape (B*K, C), got {tuple(class_logits_flat.shape)}")

            num_classes_model = class_logits_flat.size(1)
            # reshape to (B, K, C) then average over K => (B, C)
            class_logits = class_logits_flat.view(B, K, num_classes_model).mean(dim=1)  # (B, C)

            # compute per-sample loss only for labeled samples
            if mask_per_sample.sum() > 0:
                labels_gpu = labels.to(device)
                loss = criterion(class_logits[mask_per_sample], labels_gpu[mask_per_sample].to(device))
                loss_list.append(float(loss.item()))
            # else: no labeled samples in this batch — skip loss bookkeeping

            # probs & preds per-sample
            probs = torch.softmax(class_logits, dim=1).cpu().numpy()  # (B, C)
            if num_classes == 2:
                preds = (probs[:, 1] >= 0.5).astype(int).tolist()
                prob_vals = probs[:, 1].tolist()
            else:
                preds = np.argmax(probs, axis=1).tolist()
                prob_vals = probs.tolist()

            # append only labeled entries
            for i in range(B):
                if not mask_per_sample[i]:
                    continue
                all_preds.append(int(preds[i]))
                all_probs.append(prob_vals[i])
                all_trues.append(int(labels[i].item()))

    # compute final metrics
    if len(all_trues) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": None, "mcc": None}

    acc = accuracy_score(all_trues, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average="binary" if num_classes == 2 else "macro", zero_division=0)
    try:
        roc_auc = None
        if num_classes == 2 and len(set(all_trues)) > 1:
            roc_auc = float(roc_auc_score(all_trues, all_probs))
    except Exception:
        roc_auc = None

    try:
        mcc = float(matthews_corrcoef(all_trues, all_preds))
    except Exception:
        mcc = None

    mean_loss = float(np.mean(loss_list)) if loss_list else 0.0

    return {
        "loss": mean_loss,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "mcc": mcc,
        "n_samples": len(all_trues),
    }


def evaluate_ensemble(model_paths: List[str], model_builders: List[Any], dataloader: DataLoader, device: torch.device, num_classes: int, weights: List[float]=None):
    models=[]
    for i, path in enumerate(model_paths):
        ckpt = torch.load(path,map_location=device)
        m = model_builders[i](num_classes=num_classes, pretrained=False)
        m = safe_model_load(m, ckpt)
        m.to(device)
        m.eval()
        models.append(m)
    if weights is None:
        weights=[1.0]*len(models)
    weights = np.array(weights,dtype=np.float32)/sum(weights)

    all_preds, all_trues, all_probs, losses=[], [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = batch
            images = images.to(device)
            labels_tensor = torch.tensor([(-1 if l is None else l) for l in labels],dtype=torch.long)
            mask = labels_tensor>=0
            if mask.sum()==0: continue
            labels_tensor = labels_tensor.to(device)
            ensemble_probs = None
            for w, m in zip(weights, models):
                out = m(images)
                prob = torch.softmax(out, dim=1)
                ensemble_probs = w*prob if ensemble_probs is None else ensemble_probs + w*prob
            loss = criterion(ensemble_probs, labels_tensor)
            losses.append(loss.item())
            pred = torch.argmax(ensemble_probs, dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_trues.extend(labels_tensor.cpu().tolist())
            if num_classes==2: all_probs.extend(ensemble_probs[:,1].cpu().tolist())
            else: all_probs.extend(ensemble_probs.cpu().tolist())
    avg_loss = sum(losses)/len(losses) if losses else 0.0
    acc = accuracy_score(all_trues, all_preds) if all_preds else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average="binary" if num_classes==2 else "macro", zero_division=0)
    roc_auc = 0.0
    if num_classes==2:
        try: roc_auc = roc_auc_score(all_trues, all_probs)
        except: pass
    return {"loss":avg_loss,"accuracy":acc,"precision":float(precision),"recall":float(recall),"f1":float(f1),"roc_auc":float(roc_auc)}

def normalize_arg_list(x):
    if x is None:
        return []
    # If argparse already gave a list (nargs="+"), join then split to normalize mixed input
    if isinstance(x, list):
        x = " ".join(x)
    # split on comma, plus, colon or whitespace
    parts = re.split(r"[,\+: \t]+", x)
    return [p for p in parts if p]

# -------------------------
# Helpers
# -------------------------
def infer_domain_from_path(path: str, synthetic_indicators: Optional[List[str]] = None) -> int:
    """
    Heuristic: 0 -> real, 1 -> synthetic
    Checks keys (is_synthetic, domain) then path substrings
    """
    p = (path or "").lower()
    if not synthetic_indicators:
        synthetic_indicators = ("synthetic", "synth", "ai_", "t2i", "i2i", "generated", "fake")
    for s in synthetic_indicators:
        if s.lower() in p:
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

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss
