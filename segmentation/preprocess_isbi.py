"""
ISBI 2012 challenge dataset from the U-NET paper
"""
import json
import os
import shutil

import argh
import numpy as np
from PIL import Image

SOURCE_PATH = os.environ['SOURCE_DATA_PATH']
TARGET_PATH = os.environ['DATA_PATH']

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, 'annotations')
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, 'img_with_margin_0')


def process_images_in_chunks(args):
    split_key, img_ids = args
    chunk_img_ids = []

    unique_classes = set()

    for img_id in img_ids:
        chunk_img_ids.append(img_id)

        # 1. Save labels
        if split_key != 'test':
            with open(os.path.join(SOURCE_PATH, f'SegmentationClassAug/{img_id}.png'), 'rb') as f:
                img = Image.open(f).convert('RGB')

            pix = np.array(img).astype(np.uint8)
            pix = pix[:, :, 0]
            unique_classes.update(set(np.unique(pix)))
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
        # pix.shape = (height, width, channels)
        np.save(os.path.join(MARGIN_IMG_DIR, split_key, img_id), pix)

    return chunk_img_ids, unique_classes


def preprocess_isbi():
    print(f"Preprocessing ISBI 2012")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    train_ann_dir = os.path.join(ANNOTATIONS_DIR, 'train')
    test_ann_dir = os.path.join(ANNOTATIONS_DIR, 'test')
    train_img_dir = os.path.join(MARGIN_IMG_DIR, 'train')
    test_img_dir = os.path.join(MARGIN_IMG_DIR, 'test')

    os.makedirs(train_ann_dir, exist_ok=True)
    os.makedirs(test_ann_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    all_imgs = {'train': [], 'val': [], 'test': []}

    for file in os.listdir(os.path.join(SOURCE_PATH, 'images')):
        path = os.path.join(SOURCE_PATH, 'images', file)
        img_id = file.split('.')[0].split('-')[-1]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        output_img_path = os.path.join(train_img_dir, f'{img_id}.png')
        img.save(output_img_path)

        # Save image as .npy for fast loading
        pix = np.array(img).astype(np.uint8)
        # pix.shape = (height, width, channels)
        np.save(os.path.join(train_img_dir, img_id), pix)

        all_imgs['train'].append(img_id)

    for file in os.listdir(os.path.join(SOURCE_PATH, 'labels')):
        path = os.path.join(SOURCE_PATH, 'labels', file)
        img_id = file.split('.')[0].split('-')[-1]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Save image as .npy for fast loading
        pix = (np.array(img) / 255).astype(np.uint8)
        pix = pix[:, :, 0]

        # pix.shape = (height, width, channels)
        np.save(os.path.join(train_ann_dir, img_id), pix)

    for file in os.listdir(os.path.join(SOURCE_PATH, 'test_source')):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if 'predict' in file:
            pix = np.array(img).astype(np.uint8)
            img_id = file.split('.')[0].split('_')[0]

            pix = pix[:, :, 0] / 255
            np.save(os.path.join(test_ann_dir, f'{img_id}_pred'), pix)

            pix = np.round(pix).astype(np.uint8)
            np.save(os.path.join(test_ann_dir, img_id), pix)

        else:
            path = os.path.join(SOURCE_PATH, 'test_source', file)
            img_id = file.split('.')[0]
            all_imgs['val'].append(img_id)
            all_imgs['test'].append(img_id)

            output_img_path = os.path.join(test_img_dir, f'{img_id}.png')
            img.save(output_img_path)

            # Save image as .npy for fast loading
            pix = np.array(img).astype(np.uint8)
            # pix.shape = (height, width, channels)
            np.save(os.path.join(test_img_dir, img_id), pix)

    with open(os.path.join(TARGET_PATH, 'all_images.json'), 'w') as fp:
        json.dump(all_imgs, fp)


if __name__ == '__main__':
    argh.dispatch_command(preprocess_isbi)
