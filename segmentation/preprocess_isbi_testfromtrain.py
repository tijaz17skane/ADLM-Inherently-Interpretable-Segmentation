"""
ISBI 2012 challenge dataset from the U-NET paper.
Generate a custom 'test' dataset, using only train labels

"""
import json
import os

import argh
import numpy as np
from PIL import Image

SOURCE_PATH = os.environ['SOURCE_DATA_PATH']
TARGET_PATH = os.environ['DATA_PATH']

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, 'annotations')
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, 'img_with_margin_0')

N_TEST_SAMPLES = 10


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

    test_images = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

    for file in os.listdir(os.path.join(SOURCE_PATH, 'images')):
        path = os.path.join(SOURCE_PATH, 'images', file)
        img_id = file.split('.')[0].split('-')[-1]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if int(img_id) in test_images:
            output_img_path = os.path.join(test_img_dir, f'{img_id}.png')
            output_np_path = os.path.join(test_img_dir, img_id)

            all_imgs['val'].append(img_id)
            all_imgs['test'].append(img_id)
        else:
            output_img_path = os.path.join(train_img_dir, f'{img_id}.png')
            output_np_path = os.path.join(train_img_dir, img_id)

            all_imgs['train'].append(img_id)

        img.save(output_img_path)

        # Save image as .npy for fast loading
        pix = np.array(img).astype(np.uint8)
        # pix.shape = (height, width, channels)
        np.save(output_np_path, pix)

    for file in os.listdir(os.path.join(SOURCE_PATH, 'labels')):
        path = os.path.join(SOURCE_PATH, 'labels', file)
        img_id = file.split('.')[0].split('-')[-1]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if int(img_id) in test_images:
            ann_dir = test_ann_dir
        else:
            ann_dir = train_ann_dir

        # Save image as .npy for fast loading
        pix = (np.array(img) / 255).astype(np.uint8)
        pix = pix[:, :, 0]

        # pix.shape = (height, width, channels)
        np.save(os.path.join(ann_dir, img_id), pix)

    with open(os.path.join(TARGET_PATH, 'all_images.json'), 'w') as fp:
        json.dump(all_imgs, fp)


if __name__ == '__main__':
    argh.dispatch_command(preprocess_isbi)
