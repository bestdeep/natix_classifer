import os
from typing import List, Optional, Callable, Union, Dict, Tuple
import json

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# keep these imports from your project
from datasets.image_transforms import apply_augmentation_by_level, TARGET_IMAGE_SIZE
from datasets.get_paths import load_image_metadata_pairs_from_file

ImageLike = Union[np.ndarray, Image.Image]

class NatixDataset(Dataset):
    """
    Improved dataset for images with per-image JSON metadata.

    Features:
    - recursive directory walk (keeps deterministic order)
    - lazy image loading (preload_images=True to load all at init)
    - optional metadata preloading (preload_metadata=True)
    - support for custom transform (callable(image)->tensor/array)
    - optional augmentation using apply_augmentation_by_level (augment=True)
    - optional mapping from string labels to integer indices via label_map
    - label_key: which key to read from JSON metadata (default 'label')

    Args:
        dirs: list of directories to scan (recursively)
        transform: optional callable applied after augmentation. If None, images are converted to torch.FloatTensor [0..1].
        augment: if True, use apply_augmentation_by_level(image, TARGET_IMAGE_SIZE).
        label_key: key name in JSON metadata to use as label (default 'label')
        label_map: optional dict mapping label values -> ints. If not provided, one will be created from dataset metadata.
        preload_images: whether to load all images into memory at init (False by default)
        preload_metadata: whether to load all metadata JSON into memory at init (True recommended only for small datasets)
        image_extensions: tuple of allowed image extensions
    """

    def __init__(
        self,
        filelist_path: str = "",
        transform: Optional[Callable[[ImageLike], Union[torch.Tensor, np.ndarray]]] = None,
        augment: bool = True,
        label_key: str = "label",
        label_map: Optional[Dict[Union[str, int], int]] = None,
        preload_images: bool = False,
    ):
        if not filelist_path:
            raise ValueError("filelist_path must be a non-empty string.")
        self.filelist_path = filelist_path
        self.transform = transform
        self.augment = augment
        self.label_key = label_key
        self.preload_images = preload_images

        # gather files
        self.image_metadata_pairs = load_image_metadata_pairs_from_file(self.filelist_path)
        self.image_file_paths = [pair[0] for pair in self.image_metadata_pairs]
        if len(self.image_file_paths) == 0:
            raise ValueError(f"No image files found in provided list files: {filelist_path}")

        # pair metadata files deterministically (same order as images)
        self.labels = [pair[1] for pair in self.image_metadata_pairs]

        # optional preloads
        self._image_cache: Dict[int, Image.Image] = {}

        if self.preload_images:
            for i, img_path in enumerate(self.image_file_paths):
                self._image_cache[i] = Image.open(img_path).convert("RGB")

        self.label_map = label_map or self._build_label_map()

    def _load_image_pil(self, idx: int) -> Image.Image:
        if idx in self._image_cache:
            return self._image_cache[idx]
        path = self.image_file_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image {path}: {e}")
        if self.preload_images:
            self._image_cache[idx] = img
        return img

    # ----------------------
    # Label helpers
    # ----------------------
    def _build_label_map(self) -> Dict[Union[str, int], int]:
        # Build mapping from all found labels -> integer indices (deterministic)
        unique = sorted({l for l in self.labels if l is not None}, key=lambda x: str(x))
        return {v: i for i, v in enumerate(unique)}

    def _label_to_index(self, label: Union[str, int, None]) -> Optional[int]:
        if label is None:
            return None
        if label in self.label_map:
            return self.label_map[label]
        # fallback: try to coerce string-int
        try:
            return int(label)
        except Exception:
            # unseen label -> extend label_map (not recommended during training)
            new_index = len(self.label_map)
            self.label_map[label] = new_index
            return new_index

    # ----------------------
    # Dataset protocol
    # ----------------------
    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, idx: int):
        """
        Returns:
            A tuple (image_tensor, label_index, image_path)
             - image_tensor: torch.Tensor (C,H,W) float32 in [0,1] unless transform returns otherwise
             - label_index: int (or None if missing)
             - image_path: str
        """
        if idx < 0:
            idx = len(self) + idx
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        label_index = self._label_to_index(self.labels[idx])

        # load image and convert to numpy or PIL depending on augment/transform
        pil_img = self._load_image_pil(idx)

        # augment using project augmentation (returns transformed, level, params)
        if self.augment:
            # apply_augmentation_by_level expects a numpy array input in your original code.
            transformed, level, params = apply_augmentation_by_level(pil_img, TARGET_IMAGE_SIZE)
            # `transformed` may be numpy array; let transform handle conversion to tensor if provided
            img_out = transformed
        else:
            img_out = pil_img

        # final transform / ensure torch tensor [C,H,W] float32 in [0..1]
        if self.transform is not None:
            img_t = self.transform(img_out)
        else:
            # default conversion pipeline to tensor float [0..1]
            if isinstance(img_out, np.ndarray):
                arr = img_out
            else:  # PIL.Image
                arr = np.array(img_out)
            # arr shape: H,W,C
            if arr.dtype != np.uint8:
                # coerce to uint8 if possible
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            # convert to float tensor
            img_t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)

        return img_t, label_index, self.image_file_paths[idx]

    # convenience
    def class_count(self) -> Dict[Union[str, int], int]:
        counts: Dict[Union[str, int], int] = {}
        for i in range(len(self)):
            lbl = self.labels[i]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts
