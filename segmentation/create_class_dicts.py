"""
Create and save dictionaries that map classes to the images that contain them.
This allows for training a sliding window segmentation model that is balanced with respect to classes
"""
import multiprocessing
import os
from tqdm import tqdm
from settings import data_path
import numpy as np
from collections import Counter, defaultdict
import json


def save_cls2pixel_parallel(args):
    paths_chunk, output_dir, output_cls2idx_dir = args
    cls2images = defaultdict(list)
    cls_counts = Counter()

    for ann_path in paths_chunk:
        img_id = ann_path.split('.')[-2].split('/')[-1]
        ann = np.load(ann_path)
        cls2idx = {}

        for cls_num in np.unique(ann):
            str_cls = str(cls_num)
            cls2images[str_cls].append(img_id)

            is_cls = ann == cls_num
            cls_idx = np.argwhere(is_cls)

            cls2idx[str_cls] = cls_idx

            cls_counts[str_cls] += np.sum(is_cls)

        np.savez(os.path.join(output_cls2idx_dir, f'{img_id}.npz'), **cls2idx)

    return cls_counts, cls2images


def create_class_dicts(chunk_size: int = 100, n_jobs: int = -1, include_test: bool = False):
    cls_counts = Counter()

    splits = ['train', 'val'] if not include_test else ['train', 'val', 'test']

    for split_key in splits:
        cls2images = defaultdict(list)
        annotations_dir = os.path.join(data_path, 'annotations', split_key)
        output_dir = os.path.join(data_path, 'class2pixel', split_key)
        output_cls2idx_dir = os.path.join(data_path, 'class2pixel', split_key, 'cls2idx')
        os.makedirs(output_cls2idx_dir, exist_ok=True)

        all_paths = [os.path.join(annotations_dir, p) for p in os.listdir(annotations_dir) if p.endswith('.npy')]

        n_chunks = max(1, int(np.ceil(len(all_paths) / chunk_size)))
        path_chunks = np.array_split(np.asarray(all_paths), n_chunks)
        parallel_args = [(chunk, output_dir, output_cls2idx_dir) for chunk in path_chunks]

        print(f'{split_key} set - building class-to-pixel dictionaries in {n_chunks} chunks.')
        pool = multiprocessing.Pool(multiprocessing.cpu_count() if n_jobs == -1 else n_jobs)

        for chunk_cls_counts, chunk_cls2img in tqdm(pool.imap_unordered(save_cls2pixel_parallel, parallel_args),
                                                    desc=split_key, total=len(parallel_args)):
            if split_key == 'train':
                cls_counts += chunk_cls_counts

            for c, paths in chunk_cls2img.items():
                cls2images[c] = cls2images[c] + paths

        cls2images = {k: np.asarray(v) for k, v in cls2images.items()}

        print()
        print(f'{split_key} set. Image counts for each class:')
        for k in sorted(cls2images.keys(), key=int):
            print("{:3d}: {:d}".format(int(k), len(cls2images[k])))

            # image IDs should be unique
            assert len(set(cls2images[k])) == len(cls2images[k])
        print()

        np.savez(os.path.join(output_dir, 'cls2img.npz'), **cls2images)

        if split_key == 'train':
            total_cls_counts = sum(cls_counts.values())
            cls_counts = {k: float(cls_counts[k] / total_cls_counts) for k
                          in sorted(cls_counts.keys(), key=lambda k: -cls_counts[k])}

            with open(os.path.join(data_path, 'class2pixel/class_counts.json'), 'w') as fp:
                json.dump(cls_counts, fp, indent=2)


if __name__ == '__main__':
    create_class_dicts()
