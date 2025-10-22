import os
from typing import List, Optional, Callable, Union, Dict, Tuple, Iterator
import random
from collections import defaultdict

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
import math

# keep these imports from your project
from datasets.image_transforms import apply_augmentation_by_level, TARGET_IMAGE_SIZE
from datasets.get_paths import load_image_metadata_pairs_from_file
from model.utils import infer_domain_from_path

ImageLike = Union[np.ndarray, Image.Image]

class NatixDataset(Dataset):
    """
    Improved dataset for images with per-image JSON metadata.
      - lazy loading
      - domain detection (real vs synthetic)
      - returning multiple augmented versions per image via `num_augmentations`
      - deterministic ordering

    Args:
        dirs: list of directories to scan (recursively)
        transform: optional callable applied after augmentation. If None, images are converted to torch.FloatTensor [0..1].
        num_augmentations: number of augmented versions returned per sample (K). If 0 or 1, returns single augmentation (shape changed accordingly).
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
        num_augmentations: int = 1,
        augment: bool = True,
        label_key: str = "label",
        label_map: Optional[Dict[Union[str, int], int]] = None,
        preload_images: bool = False,
    ):
        if not filelist_path:
            raise ValueError("filelist_path must be a non-empty string.")
        self.filelist_path = filelist_path
        self.transform = transform
        self.num_augmentations = max(1, int(num_augmentations))
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

        self.real_indices = []
        self.synthetic_indices = []
        for i, path in enumerate(self.image_file_paths):
            is_synth = infer_domain_from_path(path)
            if is_synth == 1:
                self.synthetic_indices.append(i)
            else:
                self.real_indices.append(i)


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
          images_k: torch.Tensor (K, C, H, W) if transform produces tensor, else list of K images
          label_index: int or None
          path: str
          domain: int (0 real, 1 synthetic)
        """
        if idx < 0:
            idx = len(self) + idx
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        label_index = self._label_to_index(self.labels[idx])

        # load image and convert to numpy or PIL depending on augment/transform
        pil_img = self._load_image_pil(idx)

        # produce K augmentations
        augmented_list = []
        for k in range(self.num_augmentations):
            # augment using project augmentation (returns transformed, level, params)
            transformed = None
            if self.augment:
                transformed, level, params = apply_augmentation_by_level(pil_img, TARGET_IMAGE_SIZE)
            else:
                transformed, level, params = apply_augmentation_by_level(pil_img, TARGET_IMAGE_SIZE, level_probs={0: 1})

            # final transform / ensure torch tensor [C,H,W] float32 in [0..1]
            if self.transform is not None:
                img_t = self.transform(transformed)
            else:
                # default conversion pipeline to tensor float [0..1]
                if isinstance(transformed, np.ndarray):
                    arr = transformed
                else:  # PIL.Image
                    arr = np.array(transformed)
                # arr shape: H,W,C
                if arr.dtype != np.uint8:
                    # coerce to uint8 if possible
                    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                # convert to float tensor
                img_t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
            augmented_list.append(img_t)

        images_k = torch.stack(augmented_list, dim=0)
        path = self.image_file_paths[idx]
        domain = infer_domain_from_path(path)
        return images_k, label_index, path, domain

    # convenience
    def class_count(self) -> Dict[Union[str, int], int]:
        counts: Dict[Union[str, int], int] = {}
        for i in range(len(self)):
            lbl = self.labels[i]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

class BalancedDomainSampler(Sampler):
    """
    Creates batches containing half real and half synthetic indices.

    Usage:
      sampler = BalancedDomainSampler(real_indices, synth_indices, batch_size, shuffle=True, drop_last=True)
      loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_multi, ...)
    """
    def __init__(self, real_indices: List[int], synth_indices: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        assert batch_size % 2 == 0, "batch_size must be even for balanced sampler"
        self.real_indices = list(real_indices)
        self.synth_indices = list(synth_indices)
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.shuffle = shuffle
        self.drop_last = drop_last

        # If one side empty, fallback to sampling from the other side duplicated
        if len(self.real_indices) == 0 and len(self.synth_indices) == 0:
            raise ValueError("Both real and synthetic index lists are empty.")

    def __len__(self):
        # approximate number of samples: total // batch_size
        total = len(self.real_indices) + len(self.synth_indices)
        if self.drop_last:
            return total // self.batch_size
        else:
            return math.ceil(total / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        reals = self.real_indices.copy()
        synths = self.synth_indices.copy()
        if self.shuffle:
            random.shuffle(reals)
            random.shuffle(synths)

        # cycle iterators when one side is shorter
        real_pos = 0
        synth_pos = 0
        real_len = max(1, len(reals))
        synth_len = max(1, len(synths))

        batches = []
        # compute number of batches to yield
        if self.drop_last:
            n_batches = (len(reals) + len(synths)) // self.batch_size
        else:
            n_batches = math.ceil((len(reals) + len(synths)) / self.batch_size)

        for _ in range(n_batches):
            batch = []
            # take half from reals
            for i in range(self.half):
                if len(reals) == 0:
                    # fallback to synths only
                    idx = synths[synth_pos % synth_len]
                    synth_pos += 1
                    batch.append(idx)
                else:
                    idx = reals[real_pos % real_len]
                    real_pos += 1
                    batch.append(idx)
            # half from synths
            for i in range(self.half):
                if len(synths) == 0:
                    idx = reals[real_pos % real_len]
                    real_pos += 1
                    batch.append(idx)
                else:
                    idx = synths[synth_pos % synth_len]
                    synth_pos += 1
                    batch.append(idx)
            if self.shuffle:
                random.shuffle(batch)
            batches.append(batch)

        for b in batches:
            yield b



