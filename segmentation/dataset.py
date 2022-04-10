"""
Dataset for training prototype patch classification model on Cityscapes and SUN datasets
"""
import json
from typing import Any, List
from tqdm import tqdm

import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
import os
import gin

from settings import data_path, log

import numpy as np


@gin.configurable(allowlist=['mean', 'std', 'transpose_ann', 'image_margin_size', 'patch_size', 'num_classes'])
class PatchClassificationDataset(VisionDataset):
    def __init__(
            self,
            split_key: str,
            is_eval: bool,
            model_image_size: int,
            push_prototypes: bool = False,
            mean: List[float] = gin.REQUIRED,
            std: List[float] = gin.REQUIRED,
            transpose_ann: bool = gin.REQUIRED,
            image_margin_size: int = gin.REQUIRED,
            patch_size: float = gin.REQUIRED,
            num_classes: float = gin.REQUIRED
    ):
        self.mean = mean
        self.std = std
        self.model_image_size = model_image_size
        self.is_eval = is_eval
        self.split_key = split_key
        self.annotations_dir = os.path.join(data_path, 'annotations', split_key)
        self.push_prototypes = push_prototypes
        self.transpose_ann = transpose_ann
        self.image_margin_size = image_margin_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # we generated cityscapes images with max margin earlier
        self.img_dir = os.path.join(data_path, f'img_with_margin_{self.image_margin_size}/{split_key}')

        if push_prototypes:
            transform = None
        elif self.is_eval:
            transform = transforms.Compose([
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.Normalize(mean, std)
            ])

        super(PatchClassificationDataset, self).__init__(
            root=self.img_dir,
            transform=transform
        )

        with open(os.path.join(data_path, 'all_images.json'), 'r') as fp:
            self.img_ids = json.load(fp)[split_key]

        self.img_id2idx = {img_id: i for i, img_id in enumerate(self.img_ids)}

        if bool(int(os.environ['LOAD_IMAGES_RAM'])):
            log(f"Loading {len(self.img_ids)} samples from {split_key} set to memory...")
            self.loaded_images, self.loaded_annotations = self.load_images()
        else:
            self.loaded_images, self.loaded_annotations = None, None

        log(f"Loaded {len(self.img_ids)} samples from {split_key} set")

        self.cached_img = None
        self.cached_img_id = None

    def __len__(self) -> int:
        return len(self.img_ids)

    def get_img_path(self, img_id: str) -> str:
        return os.path.join(self.img_dir, img_id + '.png')

    def load_images(self):
        images = []
        annotations = []

        for img_id in self.img_ids:
            img_path = os.path.join(self.img_dir, img_id + '.npy')
            ann_path = os.path.join(self.annotations_dir, img_id + '.npy')

            img = np.load(img_path)
            ann = np.load(ann_path)

            images.append(img)

            if self.transpose_ann:
                ann = ann.transpose()
            annotations.append(ann)

        return images, annotations

    def _load_img_and_ann(self, img_index: int, img_id: str):
        if self.loaded_images is None:
            if self.cached_img_id == img_id and self.cached_img is not None:
                img = self.cached_img[0]
                ann = self.cached_img[1]
            else:
                self.cached_img, self.cached_img_id = None, None
                img_path = os.path.join(self.img_dir, img_id + '.npy')
                ann_path = os.path.join(self.annotations_dir, img_id + '.npy')

                img = np.load(img_path)
                ann = np.load(ann_path)

                if self.transpose_ann:
                    ann = ann.transpose()

                self.cached_img = img, ann
                self.cached_img_id = img_id
        else:
            img = self.loaded_images[img_index]
            ann = self.loaded_annotations[img_index]

        return img, ann

    def __getitem__(self, index: int) -> Any:
        try:
            img_id = self.img_ids[index]
            img, ann = self._load_img_and_ann(index, img_id)

            full_ann = np.full((img.shape[0], img.shape[1]), fill_value=-1)
            full_ann[self.image_margin_size:-self.image_margin_size,
                     self.image_margin_size:-self.image_margin_size] = ann

            # insert class for mirrored margin
            full_ann[:self.image_margin_size, :] = np.flip(
                full_ann[self.image_margin_size:2 * self.image_margin_size, :],
                axis=0)
            full_ann[-self.image_margin_size:, :] = np.flip(
                full_ann[-2 * self.image_margin_size:-self.image_margin_size, :],
                axis=0)

            full_ann[:, :self.image_margin_size] = np.flip(
                full_ann[:, self.image_margin_size:2 * self.image_margin_size],
                axis=1)
            full_ann[:, -self.image_margin_size:] = np.flip(
                full_ann[:, -2 * self.image_margin_size:-self.image_margin_size],
                axis=1)

            full_ann[:self.image_margin_size, :self.image_margin_size] = np.flip(
                full_ann[self.image_margin_size:2 * self.image_margin_size,
                         self.image_margin_size:2 * self.image_margin_size],
                axis=None)
            full_ann[-self.image_margin_size:, -self.image_margin_size:] = np.flip(
                full_ann[-2 * self.image_margin_size:-self.image_margin_size,
                         -2 * self.image_margin_size:-self.image_margin_size],
                axis=None)

            full_ann[-self.image_margin_size:, :self.image_margin_size] = np.flip(
                full_ann[-2 * self.image_margin_size:-self.image_margin_size,
                         self.image_margin_size:2 * self.image_margin_size],
                axis=None)
            full_ann[:self.image_margin_size, -self.image_margin_size:] = np.flip(
                full_ann[self.image_margin_size:2 * self.image_margin_size,
                         -2 * self.image_margin_size:-self.image_margin_size],
                axis=None)

            # assert np.sum(full_ann == -1) == 0

            if self.is_eval:
                h_shift = 0
                v_shift = 0
            else:
                # shift image randomly
                h_shift = np.random.randint(-self.patch_size + 1, self.patch_size)
                v_shift = np.random.randint(-self.patch_size + 1, self.patch_size)

            img = img[self.image_margin_size + h_shift:-self.image_margin_size + h_shift,
                      self.image_margin_size + v_shift:-self.image_margin_size + v_shift]
            target = full_ann[self.image_margin_size + h_shift:-self.image_margin_size + h_shift,
                              self.image_margin_size + v_shift: -self.image_margin_size + v_shift]

            img = torch.tensor(img).permute(2, 0, 1) / 256

            # Random horizontal flip
            if not self.is_eval and np.random.random() > 0.5:
                target = np.flip(target, axis=1)
                img = torch.flip(img, [2])

            # Get target as a distribution of classes per patch
            n_target_rows, n_target_cols = int(img.shape[1] / self.patch_size), int(img.shape[2] / self.patch_size)

            target_dist = np.full((n_target_rows, n_target_cols, self.num_classes), fill_value=0.0)

            for i in range(n_target_rows):
                for j in range(n_target_cols):
                    patch_classes = target[i * self.patch_size:(i + 1) * self.patch_size,
                                    j * self.patch_size:(j + 1) * self.patch_size]
                    pixels_in_patch = patch_classes.size
                    unique, counts = np.unique(patch_classes.flatten(), return_counts=True)
                    for n, c in zip(unique, counts):
                        target_dist[i, j, n] = c / pixels_in_patch

            target = target_dist

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        except Exception as e:
            raise e
            # TODO catch errors


if __name__ == '__main__':
    dataset = PatchClassificationDataset(
        split_key='train',
        is_eval=True,
        min_window_size=224,
        max_window_size=224,
        model_image_size=True
    )

    for i in tqdm(range(len(dataset)), desc='smoke testing dataset'):
        im, tgt = dataset[i]
