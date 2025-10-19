import os
from typing import List, Optional, Callable, Union, Dict, Tuple
import json

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# keep these imports from your project
from datasets.image_transforms import apply_augmentation_by_level, TARGET_IMAGE_SIZE

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
        dirs: List[str],
        transform: Optional[Callable[[ImageLike], Union[torch.Tensor, np.ndarray]]] = None,
        augment: bool = True,
        label_key: str = "label",
        label_map: Optional[Dict[Union[str, int], int]] = None,
        preload_images: bool = False,
        preload_metadata: bool = False,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".gif"),
    ):
        if not dirs:
            raise ValueError("dirs must be a non-empty list of directories.")
        self.dirs = dirs
        self.transform = transform
        self.augment = augment
        self.label_key = label_key
        self.preload_images = preload_images
        self.preload_metadata = preload_metadata
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        # gather files
        self.image_file_paths = self._gather_image_files(self.dirs)
        if len(self.image_file_paths) == 0:
            raise ValueError(f"No image files found in provided directories: {dirs}")

        # pair metadata files deterministically (same order as images)
        self.metadata_file_paths = self._pair_metadata_files(self.image_file_paths)

        # sanity check
        if len(self.image_file_paths) != len(self.metadata_file_paths):
            raise ValueError("Mismatch between found images and metadata files after pairing.")

        # optional preloads
        self._image_cache: Dict[int, Image.Image] = {}
        self._meta_cache: Dict[int, dict] = {}

        if self.preload_metadata:
            for i, meta_path in enumerate(self.metadata_file_paths):
                self._meta_cache[i] = self._load_metadata_file(meta_path)

        if self.preload_images:
            for i, img_path in enumerate(self.image_file_paths):
                self._image_cache[i] = Image.open(img_path).convert("RGB")

        # build or validate label_map
        self.label_map = label_map or self._build_label_map()
        # convert labels list (cached or on-demand when metadata preloaded)
        # We'll use lazy label lookup in __getitem__ so it's always consistent with metadata.

    # ----------------------
    # File gathering helpers
    # ----------------------
    def _is_image_file(self, filename: str) -> bool:
        return filename.lower().endswith(self.image_extensions)

    def _gather_image_files(self, dirs: List[str]) -> List[str]:
        files = []
        for dir_path in dirs:
            if not os.path.isdir(dir_path):
                raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
            for root, _, filenames in os.walk(dir_path):
                for fn in filenames:
                    if self._is_image_file(fn):
                        files.append(os.path.join(root, fn))
        # deterministic order
        files = sorted(files)
        return files

    def _pair_metadata_files(self, image_file_paths: List[str]) -> List[str]:
        metas = []
        missing_meta_images = []
        for img_path in image_file_paths:
            meta_path = os.path.splitext(img_path)[0] + ".json"
            if os.path.exists(meta_path):
                metas.append(meta_path)
            else:
                missing_meta_images.append(img_path)
        if missing_meta_images:
            raise ValueError(
                f"Missing metadata (.json) files for the following images (first 5 shown):\n"
                + "\n".join(missing_meta_images[:5])
            )
        return metas

    # ----------------------
    # Loading helpers
    # ----------------------
    def _load_metadata_file(self, meta_path: str) -> dict:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata file {meta_path}: {e}")

    def _get_metadata(self, idx: int) -> dict:
        if idx in self._meta_cache:
            return self._meta_cache[idx]
        meta = self._load_metadata_file(self.metadata_file_paths[idx])
        if self.preload_metadata:
            self._meta_cache[idx] = meta
        return meta

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
        labels = []
        for i in range(len(self.image_file_paths)):
            meta = self._get_metadata(i)
            val = meta.get(self.label_key, None)
            labels.append(val)
        unique = sorted({l for l in labels if l is not None}, key=lambda x: str(x))
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
            A tuple (image_tensor, label_index, metadata_dict, image_path)
             - image_tensor: torch.Tensor (C,H,W) float32 in [0,1] unless transform returns otherwise
             - label_index: int (or None if missing)
             - metadata_dict: the parsed JSON metadata
             - image_path: str
        """
        if idx < 0:
            idx = len(self) + idx
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        meta = self._get_metadata(idx)
        label_value = meta.get(self.label_key, None)
        label_index = self._label_to_index(label_value)

        # load image and convert to numpy or PIL depending on augment/transform
        pil_img = self._load_image_pil(idx)

        # augment using project augmentation (returns transformed, level, params)
        if self.augment:
            # apply_augmentation_by_level expects a numpy array input in your original code.
            transformed, level, params = apply_augmentation_by_level(pil_img, TARGET_IMAGE_SIZE)
            # `transformed` may be numpy array; let transform handle conversion to tensor if provided
            img_out = transformed
            # optionally attach augmentation info to metadata (not in-place)
            meta = dict(meta)  # copy to avoid mutation
            meta["_augment_level"] = level
            meta["_augment_params"] = params
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

        return img_t, label_index, meta, self.image_file_paths[idx]

    # convenience
    def class_count(self) -> Dict[Union[str, int], int]:
        counts: Dict[Union[str, int], int] = {}
        for i in range(len(self)):
            meta = self._get_metadata(i)
            lbl = meta.get(self.label_key, None)
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts
