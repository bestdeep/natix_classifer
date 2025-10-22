import os
from typing import List, Tuple
import json
import random

from model.utils import normalize_arg_list

image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(image_extensions)

def _gather_image_files(dirs: List[str]) -> List[str]:
    files = []
    for dir_path in dirs:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
        for root, _, filenames in os.walk(dir_path):
            for fn in filenames:
                if _is_image_file(fn):
                    files.append(os.path.join(root, fn))
    # deterministic order
    files = sorted(files)
    return files

def _pair_metadata_files(image_file_paths: List[str]) -> List[str]:
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

def _load_metadata_file(meta_path: str) -> dict:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata file {meta_path}: {e}")
    
def get_image_metadata_pairs(dirs: List[str]) -> Tuple[List[str], List[str]]:
    image_files = _gather_image_files(dirs)
    metadata_files = _pair_metadata_files(image_files)
    image_metadata_pairs = []
    for img_path, meta_path in zip(image_files, metadata_files):
        metadata = _load_metadata_file(meta_path)  # Validate metadata files
        image_metadata_pairs.append((img_path, metadata.get("label", None)))
    return image_metadata_pairs

def split_dataset(
    image_metadata_pairs: List[Tuple[str, str]], 
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    verbose: bool = False,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
    label_0_count = sum(1 for _, label in image_metadata_pairs if label == 0)
    label_1_count = sum(1 for _, label in image_metadata_pairs if label == 1)

    if verbose:
        print(f"Label 0 count: {label_0_count}, Label 1 count: {label_1_count}")

    train_label_0 = int(label_0_count * split_ratios[0])
    train_label_1 = int(label_1_count * split_ratios[0])
    val_label_0 = int(label_0_count * split_ratios[1])
    val_label_1 = int(label_1_count * split_ratios[1])
    test_label_0 = int(label_0_count * split_ratios[2])
    test_label_1 = int(label_1_count * split_ratios[2])

    if verbose:
        print(f"Train Label 0 count: {train_label_0}, Train Label 1 count: {train_label_1}")
        print(f"Validation Label 0 count: {val_label_0}, Validation Label 1 count: {val_label_1}")
        print(f"Test Label 0 count: {test_label_0}, Test Label 1 count: {test_label_1}")

    train_set = []
    val_set = []
    test_set = []
    current_label_0 = 0
    current_label_1 = 0

    random.seed(seed)
    random.shuffle(image_metadata_pairs)
    for img_metadata in image_metadata_pairs:
        if img_metadata[1] == 0 and current_label_0 < train_label_0:
            train_set.append(img_metadata)
            current_label_0 += 1
        elif img_metadata[1] == 1 and current_label_1 < train_label_1:
            train_set.append(img_metadata)
            current_label_1 += 1
        elif img_metadata[1] == 0 and current_label_0 < train_label_0 + val_label_0:
            val_set.append(img_metadata)
            current_label_0 += 1
        elif img_metadata[1] == 1 and current_label_1 < train_label_1 + val_label_1:
            val_set.append(img_metadata)
            current_label_1 += 1
        elif img_metadata[1] == 0 and current_label_0 < label_0_count:
            test_set.append(img_metadata)
            current_label_0 += 1
        elif img_metadata[1] == 1 and current_label_1 < label_1_count:
            test_set.append(img_metadata)
            current_label_1 += 1
    return train_set, val_set, test_set

def merge_datasets(
    datasets: List[List[Tuple[str, str]]]
) -> List[Tuple[str, str]]:
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    return merged

def load_image_metadata_pairs_from_file(
    filepath: str
) -> List[Tuple[str, str]]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            img_path, label = line.strip().split(",")
            pairs.append((img_path, int(label)))
    return pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gather image and metadata file paths.")
    subparsers = parser.add_subparsers(dest="command")

    get_path_parser = subparsers.add_parser("get_paths", help="Get image and metadata file paths from directories.")
    get_path_parser.add_argument("--dirs", nargs="+", required=True, help="List of directories to search for images.")
    get_path_parser.add_argument("--output", type=str, default="hf_extracted.txt", help="Output file to save image and metadata paths.")
    get_path_parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    get_path_parser.add_argument("--split_ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train/validation/Test split ratio.")

    merge_parser = subparsers.add_parser("merge_datasets", help="Merge multiple dataset files into one.")
    merge_parser.add_argument("--input_files", nargs="+", required=True, help="List of dataset files to merge.")
    merge_parser.add_argument("--output", type=str, default="merged_dataset.txt", help="Output file to save merged dataset.")
    merge_parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    if args.command == "get_paths":
        dirs = normalize_arg_list(args.dirs)

        image_metadata_pairs = get_image_metadata_pairs(dirs)

        with open(args.output, "w", encoding="utf-8") as f:
            for img_metadata in image_metadata_pairs:
                f.write(f"{img_metadata[0]},{img_metadata[1]}\n")

        if args.verbose:
            print(f"Total image-metadata pairs: {len(image_metadata_pairs)}")

        train_set, val_set, test_set = split_dataset(image_metadata_pairs, split_ratios=args.split_ratios, verbose=args.verbose)
        if args.verbose:
            print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}, Test set size: {len(test_set)}")

        with open("train_" + args.output, "w", encoding="utf-8") as f:
            for img_metadata in train_set:
                f.write(f"{img_metadata[0]},{img_metadata[1]}\n")

        with open("val_" + args.output, "w", encoding="utf-8") as f:
            for img_metadata in val_set:
                f.write(f"{img_metadata[0]},{img_metadata[1]}\n")

        with open("test_" + args.output, "w", encoding="utf-8") as f:
            for img_metadata in test_set:
                f.write(f"{img_metadata[0]},{img_metadata[1]}\n")

    elif args.command == "merge_datasets":
        all_datasets = []
        input_files = normalize_arg_list(args.input_files)
        for input_file in input_files:
            dataset = load_image_metadata_pairs_from_file(input_file)
            all_datasets.append(dataset)
            if args.verbose:
                print(f"Loaded {len(dataset)} pairs from {input_file}")
        merged_dataset = merge_datasets(all_datasets)
        with open(args.output, "w", encoding="utf-8") as f:
            for img_metadata in merged_dataset:
                f.write(f"{img_metadata[0]},{img_metadata[1]}\n")
        if args.verbose:
            print(f"Merged dataset size: {len(merged_dataset)}")
        exit(0)
    

    

    
