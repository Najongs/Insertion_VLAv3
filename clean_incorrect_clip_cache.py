import torch
from pathlib import Path
import argparse
import os

def main(args):
    """
    Scans the CLIP VLM feature cache directory to find and remove cache files
    with incorrect embedding dimensions.

    The correct dimension for embeddings from `cache_clip_vlm_features.py` is 512.
    This script identifies cache files where the embeddings have a different
    dimension (e.g., 4096 from an older 7B model) and deletes them.
    """
    cache_dir = Path(args.cache_dir)
    correct_dim = args.correct_dim
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found at '{cache_dir}'")
        return

    print(f"Scanning cache directory: '{cache_dir}'")
    print(f"Looking for cache files with embedding dimensions NOT equal to {correct_dim}.")

    files_to_delete = []
    
    # Use rglob to recursively find all .pt files
    cache_files = list(cache_dir.rglob("*.pt"))
    
    if not cache_files:
        print("No .pt cache files found.")
        return

    print(f"Found {len(cache_files)} total cache files. Checking dimensions...")

    for file_path in cache_files:
        try:
            # Load the cache file
            data = torch.load(file_path, map_location='cpu')

            # Check if the data is a tuple of two tensors (vision, text)
            if not (isinstance(data, tuple) and len(data) == 2 and 
                    isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor)):
                print(f"  - Found malformed cache file (not a tuple of 2 tensors): {file_path}")
                files_to_delete.append(file_path)
                continue

            # Check the dimension of the vision and text embeddings
            vision_embedding, text_embedding = data
            
            # The feature dimension is the last one
            if vision_embedding.shape[-1] != correct_dim or text_embedding.shape[-1] != correct_dim:
                print(f"  - Found incorrect dimension ({vision_embedding.shape[-1]}) in: {file_path}")
                files_to_delete.append(file_path)

        except Exception as e:
            print(f"  - Error processing file {file_path}: {e}")
            # Optionally, you might want to delete corrupted files as well
            # files_to_delete.append(file_path)
            continue
            
    if not files_to_delete:
        print("\nNo incorrect cache files found. Everything looks good.")
        return

    print(f"\nFound {len(files_to_delete)} cache files with incorrect dimensions.")
    
    if args.dry_run:
        print("\n[DRY RUN] The following files would be deleted:")
        for f in files_to_delete:
            print(f"  - {f}")
    else:
        user_input = input("Do you want to delete these files? (y/n): ")
        if user_input.lower() == 'y':
            deleted_count = 0
            for f in files_to_delete:
                try:
                    os.remove(f)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting file {f}: {e}")
            print(f"\nSuccessfully deleted {deleted_count} files.")
        else:
            print("\nDeletion cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up CLIP VLM cache files with incorrect dimensions."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/najo/NAS/VLA/dataset/cache/clip_vlm_features",
        help="Path to the clip_vlm_features cache directory."
    )
    parser.add_argument(
        "--correct_dim",
        type=int,
        default=512,
        help="The expected embedding dimension for correct cache files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan for files and print what would be deleted without actually deleting anything."
    )
    args = parser.parse_args()
    main(args)
