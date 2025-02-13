import os
import numpy as np
import argparse
from tqdm import tqdm  # For progress bar

def extract_slices_and_convert(input_dir, output_dir):
    """
    Extract slices from 3D scans, repeat each channel to make 3-channel deep,
    convert float -> int, and save the slices as .npy files.
    
    Args:
        input_dir (str): Path to the directory containing .npy files.
        output_dir (str): Path to the directory where processed slices will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    if not npy_files:
        print(f"No .npy files found in {input_dir}.")
        return

    # Process each .npy file
    for filename in tqdm(npy_files, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)  # Load the .npy file

        # Ensure the data is 3D (slices, height, width)
        if data.ndim != 3:
            print(f"Skipping {filename}: Expected 3D data, got {data.ndim}D.")
            continue

        # Process each slice
        for slice_idx in range(data.shape[0]):  # Iterate over slices
            slice_data = data[slice_idx, :, :]  # Extract a 2D slice

            # Repeat the slice to make it 3-channel deep
            slice_3channel = np.repeat(slice_data[:, :, np.newaxis], 3, axis=2)

            # Convert float -> int
            slice_3channel = slice_3channel.astype(np.int32)

            # Save the processed slice
            slice_filename = f"{filename.replace('.npy', '')}_slice{slice_idx:03d}.npy"
            slice_path = os.path.join(output_dir, slice_filename)
            np.save(slice_path, slice_3channel)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract slices from 3D scans, make 3-channel deep, and convert float -> int.")
    parser.add_argument("input", type=str, help="Path to the directory containing .npy files.")
    parser.add_argument("output", type=str, help="Path to the directory where processed slices will be saved.")
    
    args = parser.parse_args()

    # Extract slices and process them
    extract_slices_and_convert(args.input, args.output)

if __name__ == "__main__":
    main()
