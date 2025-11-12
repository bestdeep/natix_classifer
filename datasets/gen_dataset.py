import os
import argparse
import csv
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from typing import List, Tuple
from datasets.image_transforms import apply_augmentation_by_level, TARGET_IMAGE_SIZE
from datasets.get_paths import load_image_metadata_pairs_from_file

def generate_augmented_dataset(
    image_metadata_pairs: List[Tuple[str, str]],
    root_dir: str,
    output_dir: str,
    num_augmentations: int,
    metadata_path: str = "augmented_image_metadata.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    metadata_pairs = []

    for idx, (image_path, metadata) in enumerate(
        tqdm(image_metadata_pairs, desc="Generating augmented images"), start=1
    ):
        p = Path(image_path)
        combined = f"{p.parent.name}_{p.stem}"
        filename = p.stem
        sub_dir = os.path.join(output_dir, combined)
        os.makedirs(sub_dir, exist_ok=True)

        abs_path = os.path.join(root_dir, image_path.lstrip("/"))

        with Image.open(abs_path) as im:
            im = im.convert("RGB")

            for aug_idx in range(num_augmentations):
                aug_img, _, _ = apply_augmentation_by_level(im, TARGET_IMAGE_SIZE)

                # Convert tensor → PIL
                if isinstance(aug_img, torch.Tensor):
                    aug_img = to_pil_image(aug_img)

                aug_image_filename = f"{filename}_aug{aug_idx}.png"
                aug_image_path = os.path.join(sub_dir, aug_image_filename)
                aug_img.save(aug_image_path)

                relative_path = os.path.relpath(aug_image_path, output_dir)
                metadata_pairs.append([relative_path, metadata])

        if idx % 100 == 0:
            print(f"Processed {idx} images so far.")
            # Optional checkpoint save
            with open(metadata_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])
                writer.writerows(metadata_pairs)

    # ✅ Final CSV write
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(metadata_pairs)

    print(f"✅ Augmented dataset generation complete. Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented images")
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="dataset_files/image_metadata_pairs.txt",
        help="Path to the file containing image paths and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_dataset",
        help="Directory to save the augmented images and metadata files.",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=1,
        help="Number of augmentation levels to apply to each image.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Root directory for relative paths.",
    )
    args = parser.parse_args()

    print(f"Dataset file: {args.dataset_file}")
    print(f"Root directory: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of augmentations per image: {args.num_augmentations}")

    p = Path(args.dataset_file)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")
    
    augmented_path = p.stem + "_augmented.txt"
    augmented_path = os.path.join(p.parent, augmented_path)
    print(f"Augmented dataset file will be saved to: {augmented_path}")
    

    image_metadata_pairs = load_image_metadata_pairs_from_file(args.dataset_file)
    print(f"Loaded {len(image_metadata_pairs)} image-metadata pairs from {args.dataset_file}")
    print(f"Sample pair: {image_metadata_pairs[0]}")

    print("Starting augmentation process...")

    generate_augmented_dataset(
        image_metadata_pairs,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
    )