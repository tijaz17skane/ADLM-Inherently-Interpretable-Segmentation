"""
Dataset for training prototype patch classification model on Cityscapes and SUN datasets
"""
import json
from typing import Any, List, Optional, Tuple

from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
import os
import gin

from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES
from settings import data_path, log

import numpy as np


def resize_label(label, size):
    """
    Downsample labels by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    label = Image.fromarray(label.astype(float)).resize(size, resample=Image.NEAREST)
    return torch.LongTensor(np.asarray(label))


@gin.configurable(allowlist=['mean', 'std', 'image_margin_size', 'window_size', 'only_19_from_cityscapes'])
class PatchClassificationDataset(VisionDataset):
    def __init__(
            self,
            split_key: str,
            is_eval: bool,
            push_prototypes: bool = False,
            mean: List[float] = gin.REQUIRED,
            std: List[float] = gin.REQUIRED,
            image_margin_size: int = gin.REQUIRED,
            window_size: Optional[Tuple[float, float]] = None,
            only_19_from_cityscapes: bool = False

    ):
        self.mean = mean
        self.std = std
        self.is_eval = is_eval
        self.split_key = split_key
        self.annotations_dir = os.path.join(data_path, 'annotations', split_key)
        self.push_prototypes = push_prototypes
        self.image_margin_size = image_margin_size
        self.window_size = window_size
        self.only_19_from_cityscapes = only_19_from_cityscapes

        if self.only_19_from_cityscapes:
            self.convert_targets = np.vectorize(CITYSCAPES_19_EVAL_CATEGORIES.get)
        else:
            self.convert_targets = None

        # we generated cityscapes images with max margin earlier
        self.img_dir = os.path.join(data_path, f'img_with_margin_{self.image_margin_size}/{split_key}')

        if push_prototypes:
            transform = None
        else:
            transform = transforms.Compose([
                transforms.Normalize(mean, std)
            ])

        super(PatchClassificationDataset, self).__init__(
            root=self.img_dir,
            transform=transform
        )

        with open(os.path.join(data_path, 'all_images.json'), 'r') as fp:
            self.img_ids = json.load(fp)[split_key]

        self.img_id2idx = {img_id: i for i, img_id in enumerate(self.img_ids)}

        log(f"Loaded {len(self.img_ids)} samples from {split_key} set")

    def __len__(self) -> int:
        return len(self.img_ids)

    def get_img_path(self, img_id: str) -> str:
        return os.path.join(self.img_dir, img_id + '.png')

    def __getitem__(self, index: int) -> Any:
        img_id = self.img_ids[index]
        img_path = os.path.join(self.img_dir, img_id + '.npy')
        ann_path = os.path.join(self.annotations_dir, img_id + '.npy')
        img = np.load(img_path)
        ann = np.load(ann_path)

        if self.convert_targets is not None:
            ann = self.convert_targets(ann)

        full_ann = np.full((img.shape[0], img.shape[1]), fill_value=-1)

        if self.image_margin_size != 0:
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

            img = img[self.image_margin_size:-self.image_margin_size,
                      self.image_margin_size:-self.image_margin_size]
            target = full_ann[self.image_margin_size:-self.image_margin_size,
                              self.image_margin_size:-self.image_margin_size]
        else:
            target = ann

        img = torch.tensor(img).permute(2, 0, 1) / 256

        if not self.is_eval:
            scale = np.random.uniform(0.5, 1.5)
            window_size = int(np.round(scale * self.window_size[0])), int(np.round(scale * self.window_size[1]))
            window_size = min(window_size[0], target.shape[0]), min(window_size[1], target.shape[1])
        else:
            window_size = self.window_size

        window_size = min(window_size[0], target.shape[0]), min(window_size[1], target.shape[1])

        window_left = np.random.randint(0, target.shape[0] - window_size[0]+1)
        window_top = np.random.randint(0, target.shape[1] - window_size[1]+1)
        window_right = window_left + window_size[0]
        window_bottom = window_top + window_size[1]

        img = img[:, window_left:window_right, window_top:window_bottom]
        target = target[window_left:window_right, window_top:window_bottom]

        # Random horizontal flip
        if not self.is_eval and np.random.random() > 0.5:
            target = np.flip(target, axis=1)
            img = torch.flip(img, [2])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if img.shape[1] != self.window_size[0] or img.shape[2] != self.window_size[1]:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.window_size[0], self.window_size[1]),
                                                  mode='bilinear', align_corners=False)[0]

        # TODO un-hardcode size
        target = resize_label(target, size=(65, 65))

        return img, target
