import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# Step 1: Convert .nii files to .npy
def convert_nii_to_npy(source_dir, dest_dir):
    """
    Convert .nii files to .npy for quicker manipulation.
    """
    os.makedirs(dest_dir, exist_ok=True)
    nii_files = [f for f in os.listdir(source_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    for filename in tqdm(nii_files, desc="Converting .nii to .npy"):
        nii_path = os.path.join(source_dir, filename)
        nii_image = nib.load(nii_path)
        data = nii_image.get_fdata()
        npy_path = os.path.join(dest_dir, filename.replace(".nii.gz", ".npy").replace(".nii", ".npy"))
        np.save(npy_path, data)

# Step 2: Slice the 3D scans into 2D slices
def slice_scans(input_dir, output_dir):
    """
    Slice 3D scans into 2D slices and save them as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    for filename in tqdm(npy_files, desc="Slicing scans"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)
        for slice_idx in range(data.shape[2]):  # Iterate over slices (assuming z-axis is the slice dimension)
            slice_data = data[:, :, slice_idx]
            slice_filename = f"{filename.replace('.npy', '')}_slice{slice_idx:03d}.npy"
            slice_path = os.path.join(output_dir, slice_filename)
            np.save(slice_path, slice_data)

# Step 3: Normalize image slices to 0-255
def normalize_slices(input_dir):
    """
    Normalize image slices to values between 0 and 255.
    """
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    for filename in tqdm(npy_files, desc="Normalizing slices"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)
        data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        np.save(file_path, data_normalized)

# Step 4: Upsample slices to 2048x1024
def upsample_slices(input_dir):
    """
    Upsample slices to 2048x1024 using interpolation.
    """
    from skimage.transform import resize
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    for filename in tqdm(npy_files, desc="Upsampling slices"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)
        data_upsampled = resize(data, (1024, 2048), order=1, mode="constant", preserve_range=True).astype(np.uint8)
        np.save(file_path, data_upsampled)

# Step 5: Repeat image channels to make 2048x1024x3
def repeat_channels(input_dir):
    """
    Repeat image channels to make 2048x1024x3.
    """
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    for filename in tqdm(npy_files, desc="Repeating channels"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)
        data_3channel = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        np.save(file_path, data_3channel)

# Step 6: Convert .npy to .png
def convert_to_png(input_dir):
    """
    Convert .npy files to .png and save both in the same folder.
    """
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    for filename in tqdm(npy_files, desc="Converting to PNG"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)
        image = Image.fromarray(data.squeeze())
        png_path = os.path.join(input_dir, filename.replace(".npy", ".png"))
        image.save(png_path)

# Step 7 & 8: Split data into train, val, test and organize into folders
def split_and_organize(images_dir, labels_dir, dest_root, train_ratio=0.63, val_ratio=0.26, test_ratio=0.11):
    """
    Split data into train, val, test and organize into annotations and img_with_margin_0 folders.
    """
    # Create destination directories
    dest_annotations = os.path.join(dest_root, "annotations")
    dest_img_with_margin_0 = os.path.join(dest_root, "img_with_margin_0")

    for folder in [dest_annotations, dest_img_with_margin_0]:
        for subfolder in ["train", "val", "test"]:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    # Get list of files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".npy")])

    # Split the data
    train_files, test_files = train_test_split(image_files, test_size=test_ratio, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    # Function to copy files
    def copy_files(files, split_name):
        for file in files:
            # Copy images
            shutil.copy(
                os.path.join(images_dir, file),
                os.path.join(dest_img_with_margin_0, split_name, file)
            )
            shutil.copy(
                os.path.join(images_dir, file.replace(".npy", ".png")),
                os.path.join(dest_img_with_margin_0, split_name, file.replace(".npy", ".png"))
            )
            # Copy labels
            shutil.copy(
                os.path.join(labels_dir, file),
                os.path.join(dest_annotations, split_name, file)
            )

    # Copy files to respective folders
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process pancreas scans and organize into train, val, test splits.")
    parser.add_argument("source_folder", type=str, help="Path to the folder containing ImagesTr and LabelsTr.")
    parser.add_argument("dest_root", type=str, help="Path to the root directory where the processed data will be saved.")
    
    args = parser.parse_args()

    # Step 1: Convert .nii to .npy
    convert_nii_to_npy(os.path.join(args.source_folder, "ImagesTr"), os.path.join(args.dest_root, "temp_images"))
    convert_nii_to_npy(os.path.join(args.source_folder, "LabelsTr"), os.path.join(args.dest_root, "temp_labels"))

    # Step 2: Slice scans
    slice_scans(os.path.join(args.dest_root, "temp_images"), os.path.join(args.dest_root, "sliced_images"))
    slice_scans(os.path.join(args.dest_root, "temp_labels"), os.path.join(args.dest_root, "sliced_labels"))

    # Step 3: Normalize image slices
    normalize_slices(os.path.join(args.dest_root, "sliced_images"))

    # Step 4: Upsample slices
    upsample_slices(os.path.join(args.dest_root, "sliced_images"))
    upsample_slices(os.path.join(args.dest_root, "sliced_labels"))

    # Step 5: Repeat channels
    repeat_channels(os.path.join(args.dest_root, "sliced_images"))

    # Step 6: Convert to PNG
    convert_to_png(os.path.join(args.dest_root, "sliced_images"))

    # Step 7 & 8: Split and organize
    split_and_organize(
        os.path.join(args.dest_root, "sliced_images"),
        os.path.join(args.dest_root, "sliced_labels"),
        args.dest_root
    )

    print("Processing complete!")

if __name__ == "__main__":
    main()
