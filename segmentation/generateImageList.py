import os
import json
from tqdm import tqdm

def generate_image_list(img_with_margin_0_dir, output_json_path):
    """
    Generate a JSON file containing lists of image names for train, val, and test sets.
    
    Args:
        img_with_margin_0_dir (str): Path to the img_with_margin_0 folder.
        output_json_path (str): Path to save the output JSON file.
    """
    # Initialize dictionary to store image names
    image_dict = {"train": [], "val": [], "test": []}

    # Iterate through train, val, and test subfolders
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(img_with_margin_0_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Skipping.")
            continue

        # Get all .npy and .png files in the split folder
        files = [f for f in os.listdir(split_dir) if f.endswith(".npy") or f.endswith(".png")]
        # Extract unique filenames without extensions
        unique_names = sorted(list(set([os.path.splitext(f)[0] for f in files)))

        # Add to the dictionary
        image_dict[split] = unique_names

    # Save the dictionary as a JSON file
    with open(output_json_path, "w") as f:
        json.dump(image_dict, f, indent=4)

    print(f"Image list saved to {output_json_path}")

def main():
    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Generate a JSON file listing all images in img_with_margin_0.")
    parser.add_argument("img_with_margin_0_dir", type=str, help="Path to the img_with_margin_0 folder.")
    parser.add_argument("output_json_path", type=str, help="Path to save the output JSON file.")
    
    args = parser.parse_args()

    # Generate the image list
    generate_image_list(args.img_with_margin_0_dir, args.output_json_path)

if __name__ == "__main__":
    main()
