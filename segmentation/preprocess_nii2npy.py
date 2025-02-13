import os
import numpy as np
import nibabel as nib
import argparse

def convert_nii_to_npy(source_dir, dest_dir):
    """
    Convert NIfTI files from imagesTr and labelsTr directories to NumPy arrays
    and save them in the corresponding directories in the destination folder.
    
    Args:
        source_dir (str): Path to the source directory containing imagesTr and labelsTr.
        dest_dir (str): Path to the destination directory where NumPy files will be saved.
    """
    # Define the subdirectories to process
    subdirs = ["imagesTr", "labelsTr"]

    for subdir in subdirs:
        source_subdir = os.path.join(source_dir, subdir)
        dest_subdir = os.path.join(dest_dir, subdir)

        # Ensure the destination subdirectory exists
        os.makedirs(dest_subdir, exist_ok=True)

        # Check if the source subdirectory exists
        if not os.path.exists(source_subdir):
            print(f"Warning: {source_subdir} does not exist. Skipping.")
            continue

        # Iterate through all files in the source subdirectory
        for filename in os.listdir(source_subdir):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):  # Check for NIfTI files
                # Load the NIfTI file
                nii_path = os.path.join(source_subdir, filename)
                nii_image = nib.load(nii_path)
                data = nii_image.get_fdata()  # Convert to NumPy array

                # Save the NumPy array
                npy_filename = filename.replace(".nii.gz", ".npy").replace(".nii", ".npy")
                npy_path = os.path.join(dest_subdir, npy_filename)
                np.save(npy_path, data)

                print(f"Converted {filename} to {npy_filename} in {dest_subdir}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert NIfTI files in imagesTr and labelsTr to NumPy arrays.")
    parser.add_argument("source", type=str, help="Path to the source directory containing imagesTr and labelsTr.")
    parser.add_argument("dest", type=str, help="Path to the destination directory to save NumPy files.")
    
    args = parser.parse_args()

    # Convert NIfTI files to NumPy arrays
    convert_nii_to_npy(args.source, args.dest)

if __name__ == "__main__":
    main()
