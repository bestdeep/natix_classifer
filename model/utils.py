import numpy as np
import random
from typing import List, Any
from PIL import Image
import torch
import torchvision.transforms as T
from datasets.natix_dataset import NatixDataset
import torch.nn as nn
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from PIL import Image
from torch.utils.data import DataLoader, Subset
import re
from tqdm import tqdm


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
       metas: list of dicts
       paths: list of strings
    """
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    paths = [b[3] for b in batch]
    labels_t = torch.tensor(labels, dtype=torch.long)
    return imgs, labels_t, metas, paths

def filter_real_only_indices(dataset: NatixDataset, is_synthetic_key: str="is_synthetic") -> List[int]:
    idxs = []
    for i in range(len(dataset)):
        meta = dataset._get_metadata(i)
        val = meta.get(is_synthetic_key,None)
        if val is None or not bool(val): idxs.append(i)
    return idxs

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


def evaluate(model, dataloader, device, num_classes, log_image_count: int=16):
    model.eval()
    losses, preds, probs, trues = [], [], [], []
    samples, misclassified = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="val", leave=False)
        for images, labels, metas, paths in pbar:
            images = images.to(device)
            labels_tensor = torch.tensor([(-1 if l is None else l) for l in labels],dtype=torch.long)
            mask = labels_tensor>=0
            if mask.sum()==0: continue
            labels_tensor = labels_tensor.to(device)
            out = model(images)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            loss = criterion(out, labels_tensor)
            losses.append(loss.item())
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1).tolist()
            preds.extend(pred)
            probs.extend(prob[:,1].tolist() if num_classes==2 else prob.tolist())
            trues.extend(labels_tensor.cpu().tolist())
            # sample images
            for i in range(images.size(0)):
                if len(samples)<log_image_count: samples.append((images[i].cpu(), int(pred[i]), int(labels_tensor.cpu()[i].item()), paths[i]))
                if pred[i]!=int(labels_tensor.cpu()[i].item()) and len(misclassified)<log_image_count:
                    misclassified.append((images[i].cpu(), int(pred[i]), int(labels_tensor.cpu()[i].item()), paths[i]))
    if not preds:
        return {"loss":0.0,"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0,"roc_auc":0.0,"samples":samples,"misclassified":misclassified}
    avg_loss = float(sum(losses)/len(losses))
    acc = accuracy_score(trues,preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues,preds,average="binary" if num_classes==2 else "macro",zero_division=0)
    roc_auc = 0.0
    if num_classes==2:
        try: roc_auc = float(roc_auc_score(trues, probs))
        except: pass
    return {"loss":avg_loss,"accuracy":acc,"precision":float(precision),"recall":float(recall),"f1":float(f1),"roc_auc":float(roc_auc),"samples":samples,"misclassified":misclassified}

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
            images, labels, metas, paths = batch
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
