import os
import numpy as np
from PIL import Image  # For saving as PNG
import argparse
from tqdm import tqdm  # For progress bar

def normalize_and_convert(input_dir):
    """
    Normalize sliced .npy images to values between 0 and 255,
    convert them to PNG format, and save both formats in the same folder.
    
    Args:
        input_dir (str): Path to the directory containing sliced .npy files.
    """
    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    if not npy_files:
        print(f"No .npy files found in {input_dir}.")
        return

    # Process each .npy file
    for filename in tqdm(npy_files, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)  # Load the .npy file

        # Normalize the data to 0-255
        data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

        # Save the normalized .npy file
        normalized_npy_path = os.path.join(input_dir, filename.replace(".npy", "_normalized.npy"))
        np.save(normalized_npy_path, data_normalized)

        # Convert to PNG and save
        png_path = os.path.join(input_dir, filename.replace(".npy", ".png"))
        image = Image.fromarray(data_normalized.squeeze())  # Remove extra dimensions if any
        image.save(png_path)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Normalize sliced .npy images to 0-255, convert to PNG, and save both formats.")
    parser.add_argument("input", type=str, help="Path to the directory containing sliced .npy files.")
    
    args = parser.parse_args()

    # Normalize and convert to PNG
    normalize_and_convert(args.input)

if __name__ == "__main__":
    main()
