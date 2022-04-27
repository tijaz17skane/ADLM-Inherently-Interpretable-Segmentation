"""
Preprocess SUN RGB-D before training a segmentation model.
https://rgbd.cs.princeton.edu/

how to run run:

python -m sun.preprocess preprocess-sun
python -m sun.preprocess preprocess-sun-obj-masks
"""
from collections import Counter

import argh
import scipy.io

import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import json

# For SUN, we use conservative maximum margin of 112 (for a model with window size 224)
from segmentation.utils import add_margins_to_image
from segmentation.constants import SUN_CATEGORIES, sun_convert_categories, SUN_LABEL_2_ID

MARGIN_SIZE = 112

SOURCE_PATH = os.environ['SOURCE_DATA_PATH']
TARGET_PATH = os.environ['DATA_PATH']

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, 'annotations')
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, f'img_with_margin_{MARGIN_SIZE}')


def preprocess_sun():
    # this scripts about 30 minutes using a single CPU
    print(f"Using {len(SUN_CATEGORIES)} object categories")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {
        'train': [], 'val': [], 'test': []
    }

    split_mat_path = os.path.join(SOURCE_PATH, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')

    split_mat = scipy.io.loadmat(split_mat_path)

    paths = {
        'train': split_mat['trainvalsplit'][0][0][0][:, 0],
        'val': split_mat['trainvalsplit'][0][0][1][:, 0],
        'test': split_mat['alltest'][0]
    }

    for split_key in ['train', 'val', 'test']:
        class_counter = Counter()
        split_paths = paths[split_key]
        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        for sample_dir in tqdm(split_paths, desc=split_key):
            sample_dir = sample_dir[0]
            sample_dir = os.path.relpath(sample_dir, '/n/fs/sun3d/data')
            img_id = sample_dir.replace(os.sep, '_').replace('-', '_')
            sample_dir = os.path.join(SOURCE_PATH, sample_dir)

            # 1. annotations
            mat = scipy.io.loadmat(os.path.join(sample_dir, 'seg.mat'))
            names = ['void'] + list(np.concatenate([np.concatenate(n.flatten()) for n in mat['names']]))
            seg_mat = mat['seglabel'].transpose()

            names = [sun_convert_categories.get(n, n) for n in names]
            label_ids = np.zeros_like(seg_mat, dtype=np.uint8)

            for i in np.unique(seg_mat):
                name = names[i]
                name = sun_convert_categories.get(name, name)
                if name in SUN_LABEL_2_ID:
                    label_ids[seg_mat == i] = SUN_LABEL_2_ID[name]
                    class_counter[name] += 1
            np.save(os.path.join(ANNOTATIONS_DIR, split_key, f'{img_id}.npy'), label_ids)

            # 2. image with added margin
            img_dir = os.path.join(sample_dir, 'image')
            assert len(os.listdir(img_dir)) == 1
            img_path = os.listdir(img_dir)[0]
            full_img_path = os.path.join(img_dir, img_path)
            with open(full_img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')

            img_with_margin = add_margins_to_image(img, MARGIN_SIZE)
            output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, img_id + '.png')
            img_with_margin.save(output_img_path)

            img_ids[split_key].append(img_id)

        print()
        print(f"Object class counts on {split_key} set:")
        for cls, count in class_counter.most_common():
            print('{:5d}: {:s}'.format(count, cls))
        print()

    with open(os.path.join(TARGET_PATH, 'all_images.json'), 'w') as fp:
        json.dump(img_ids, fp)


def preprocess_sun_obj_masks():
    print(f"Using {len(SUN_CATEGORIES)} object categories")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    split_mat_path = os.path.join(SOURCE_PATH, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')

    split_mat = scipy.io.loadmat(split_mat_path)

    paths = {
        'train': split_mat['trainvalsplit'][0][0][0][:, 0],
        'val': split_mat['trainvalsplit'][0][0][1][:, 0],
        'test': split_mat['alltest'][0]
    }

    for split_key in ['train', 'val', 'test']:
        objects_per_image = []
        split_paths = paths[split_key]
        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        for sample_dir in tqdm(split_paths, desc=split_key):
            sample_dir = sample_dir[0]
            sample_dir = os.path.relpath(sample_dir, '/n/fs/sun3d/data')
            img_id = sample_dir.replace(os.sep, '_').replace('-', '_')
            sample_dir = os.path.join(SOURCE_PATH, sample_dir)

            mat = scipy.io.loadmat(os.path.join(sample_dir, 'seg.mat'))

            seg_mat = mat['seglabel'].transpose()

            objects_per_image.append(len(np.unique(seg_mat)))

            np.save(os.path.join(ANNOTATIONS_DIR, split_key, f'{img_id}_obj_mask.npy'), seg_mat)

        print("{:s} set. Average objects per image: {:.2f}".format(split_key, np.mean(objects_per_image)))


if __name__ == '__main__':
    argh.dispatch_commands([preprocess_sun, preprocess_sun_obj_masks])
