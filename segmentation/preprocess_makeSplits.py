import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(source_images, source_labels, dest_root, train_ratio=0.63, val_ratio=0.26, test_ratio=0.11):
    """
    Split data into train, val, and test sets and organize into the desired structure.
    
    Args:
        source_images (str): Path to the source ImagesTr folder.
        source_labels (str): Path to the source LabelsTr folder.
        dest_root (str): Path to the root directory where the split data will be saved.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, val, and test ratios must sum to 1.")

    # Create destination directories
    dest_annotations = os.path.join(dest_root, "annotations")
    dest_img_with_margin_0 = os.path.join(dest_root, "img_with_margin_0")

    for folder in [dest_annotations, dest_img_with_margin_0]:
        for subfolder in ["train", "val", "test"]:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    # Get list of files in ImagesTr and LabelsTr
    image_files = sorted([f for f in os.listdir(source_images) if f.endswith(".npy")])
    label_files = sorted([f for f in os.listdir(source_labels) if f.endswith(".npy")])

    # Ensure the files match
    if len(image_files) != len(label_files):
        raise ValueError("Mismatch between the number of images and labels.")
    for img, lbl in zip(image_files, label_files):
        if img != lbl:
            raise ValueError(f"Mismatch between image and label filenames: {img} vs {lbl}.")

    # Split the data
    train_files, test_files = train_test_split(image_files, test_size=test_ratio, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    # Function to copy files
    def copy_files(files, split_name):
        for file in files:
            # Copy images
            shutil.copy(
                os.path.join(source_images, file),
                os.path.join(dest_img_with_margin_0, split_name, file)
            )
            # Copy labels
            shutil.copy(
                os.path.join(source_labels, file),
                os.path.join(dest_annotations, split_name, file)
            )

    # Copy files to respective folders
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print("Data split and organized successfully!")

def main():
    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Split data into train, val, and test sets.")
    parser.add_argument("source_images", type=str, help="Path to the source ImagesTr folder.")
    parser.add_argument("source_labels", type=str, help="Path to the source LabelsTr folder.")
    parser.add_argument("dest_root", type=str, help="Path to the root directory where the split data will be saved.")
    
    args = parser.parse_args()

    # Perform the split
    split_data(args.source_images, args.source_labels, args.dest_root)

if __name__ == "__main__":
    main()
