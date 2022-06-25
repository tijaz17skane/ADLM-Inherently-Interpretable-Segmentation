"""
Preprocesss PASCAL VOC 2012 dataset before training a segmentation model.
https://www.cityscapes-dataset.com/

how to run run:

python -m cityscapes.preprocess preprocess-pascal {N_JOBS}
"""
import argh

import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import multiprocessing

SOURCE_PATH = os.environ['SOURCE_DATA_PATH']
TARGET_PATH = os.environ['DATA_PATH']
LABELS_PATH = os.path.join(SOURCE_PATH, 'gtFine_trainvaltest/gtFine/')
IMAGES_PATH = os.path.join(SOURCE_PATH, 'leftImg8bit_trainvaltest/leftImg8bit/')

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, 'annotations')
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, 'img_with_margin_0')


def process_images_in_chunks(args):
    split_key, img_ids = args
    chunk_img_ids = []

    for img_id in img_ids:
        img_id = img_id.split('_gtFine_labelIds.png')[0]
        chunk_img_ids.append(img_id)

        # 1. Save labels
        with open(os.path.join(SOURCE_PATH, f'SegmentationClass/{img_id}.png'), 'rb') as f:
            img = Image.open(f).convert('RGB')

        pix = np.array(img).astype(np.uint8)
        print("ANN", pix.shape, np.unique(pix))
        # pix.shape = (height, width, channels)
        np.save(os.path.join(ANNOTATIONS_DIR, split_key, img_id), pix)

        # 2. Save image
        input_img_path = os.path.join(SOURCE_PATH, f'JPEGImages/{img_id}.jpg')

        with open(input_img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, img_id + '.png')
        img.save(output_img_path)

        # Save image as .npy for fast loading
        pix = np.array(img).astype(np.uint8)
        print("IMG", pix.shape)
        # pix.shape = (height, width, channels)
        np.save(os.path.join(MARGIN_IMG_DIR, split_key, img_id), pix)

    return chunk_img_ids


def process_obj_masks_in_chunks(args):
    split_key, city_name, png_files = args

    split_dir = os.path.join(LABELS_PATH, split_key)
    city_dir = os.path.join(split_dir, city_name)

    for file in png_files:
        img_id = file.split('_gtFine_instanceIds.png')[0]

        # Save object mask labels
        with open(os.path.join(city_dir, file), 'rb') as f:
            # noinspection PyTypeChecker
            obj_ids = np.array(Image.open(f).convert('RGB'))[:, :, 0].astype(np.uint8)
        np.save(os.path.join(ANNOTATIONS_DIR, split_key, f'{img_id}_obj_mask.npy'), obj_ids)

    return len(png_files)


def preprocess_pascal(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Preprocessing PASCAL VOC 2012")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {
        'train': [], 'val': [], 'test': []
    }

    split_info_dir = os.path.join(SOURCE_PATH, 'ImageSets/Segmentation')

    for split_key in tqdm(['train', 'val', 'test'], desc='preprocessing images'):
        split_img_ids = [img_id.strip() for img_id in open(os.path.join(split_info_dir, f'{split_key}.txt'))]

        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        n_chunks = int(np.ceil(len(split_img_ids) / chunk_size))
        chunk_files = np.array_split(split_img_ids, n_chunks)

        parallel_args = [(split_key, chunk) for chunk in chunk_files]

        pool = multiprocessing.Pool(n_jobs)
        prog_bar = tqdm(total=len(split_img_ids), desc=f'{split_key}')

        for chunk_img_ids in pool.imap_unordered(process_images_in_chunks, parallel_args):
            img_ids[split_key] += chunk_img_ids
            prog_bar.update(len(chunk_img_ids))

        prog_bar.close()
        pool.close()

    with open(os.path.join(TARGET_PATH, 'all_images.json'), 'w') as fp:
        json.dump(img_ids, fp)


if __name__ == '__main__':
    argh.dispatch_commands(preprocess_pascal)
