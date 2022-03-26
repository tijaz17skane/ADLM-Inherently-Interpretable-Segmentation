"""
Create and save dictionaries that map classes to the images that contain them.
This allows for training a sliding window segmentation model that is balanced with respect to classes
"""
import multiprocessing
import os

import argh
from tqdm import tqdm
from settings import data_path
import numpy as np
from collections import Counter, defaultdict
import json


def get_cls2img_parallel(args):
    paths_chunk, split_key = args
    cls2images = defaultdict(list)
    cls_counts = Counter()

    for ann_path in paths_chunk:
        img_id = '.'.join(ann_path.split('.')[:-1]).split('/')[-1]
        ann = np.load(ann_path)

        for cls_num in np.unique(ann):
            str_cls = str(cls_num)
            cls2images[str_cls].append(img_id)

            if split_key == 'train':
                cls_counts[str_cls] += np.sum(ann == cls_num)

    return cls_counts, cls2images


def create_class_dicts(chunk_size: int = 100, n_jobs: int = -1, include_test: bool = False):
    cls_counts = Counter()

    splits = ['train', 'val'] if not include_test else ['train', 'val', 'test']

    for split_key in splits:
        cls2images = defaultdict(list)
        annotations_dir = os.path.join(data_path, 'annotations', split_key)
        output_dir = os.path.join(data_path, 'class2images', split_key)
        os.makedirs(output_dir, exist_ok=True)

        all_paths = [os.path.join(annotations_dir, p) for p in os.listdir(annotations_dir) if p.endswith('.npy')]

        n_chunks = max(1, int(np.ceil(len(all_paths) / chunk_size)))
        path_chunks = np.array_split(np.asarray(all_paths), n_chunks)
        parallel_args = [(chunk, split_key) for chunk in path_chunks]

        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(f'{split_key} set - building class-to-image dictionary in {n_chunks} chunks using {n_jobs} processes.')
        pool = multiprocessing.Pool(n_jobs)

        for chunk_cls_counts, chunk_cls2img in tqdm(pool.imap_unordered(get_cls2img_parallel, parallel_args),
                                                    desc=split_key, total=len(parallel_args)):
            if split_key == 'train':
                cls_counts += chunk_cls_counts

            for c, paths in chunk_cls2img.items():
                cls2images[c] = cls2images[c] + paths

        print()
        print(f'{split_key} set. Image counts for each class:')
        for k in sorted(cls2images.keys(), key=int):
            print("{:3d}: {:d}".format(int(k), len(cls2images[k])))

            # image IDs should be unique
            assert len(set(cls2images[k])) == len(cls2images[k])
        print()

        with open(os.path.join(output_dir, 'cls2img.json'), 'w') as fp:
            json.dump(cls2images, fp)

        if split_key == 'train':
            total_cls_counts = sum(cls_counts.values())
            cls_counts = {k: float(cls_counts[k] / total_cls_counts) for k
                          in sorted(cls_counts.keys(), key=lambda k: -cls_counts[k])}

            with open(os.path.join(data_path, 'class2images/class_counts.json'), 'w') as fp:
                json.dump(cls_counts, fp, indent=2)


if __name__ == '__main__':
    argh.dispatch_command(create_class_dicts)
