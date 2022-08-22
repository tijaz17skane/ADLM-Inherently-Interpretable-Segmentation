"""
Dataset for training prototype patch classification model on Cityscapes and SUN datasets
"""
import json
from typing import Any, List, Optional, Tuple

import cv2
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
import os
import gin
import random

from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES, PASCAL_ID_MAPPING
from settings import data_path, log

import numpy as np

ISBI_MULTIPLIER = 10


def resize_label(label, size):
    """
    Downsample labels by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    label = Image.fromarray(label.astype(float)).resize(size, resample=Image.NEAREST)
    return torch.LongTensor(np.asarray(label))


@gin.configurable(allowlist=['mean', 'std', 'image_margin_size',
                             'window_size', 'only_19_from_cityscapes',
                             'scales'])
class PatchClassificationDataset(VisionDataset):
    def __init__(
            self,
            split_key: str,
            is_eval: bool,
            push_prototypes: bool = False,
            mean: List[float] = gin.REQUIRED,
            std: List[float] = gin.REQUIRED,
            image_margin_size: int = gin.REQUIRED,
            window_size: Optional[Tuple[int, int]] = None,
            only_19_from_cityscapes: bool = False,
            scales: Tuple[int] = (1.0, ),
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
        self.scales = scales

        if self.only_19_from_cityscapes:
            self.convert_targets = np.vectorize(CITYSCAPES_19_EVAL_CATEGORIES.get)
        # isbi
        elif 'isbi' in data_path:
            self.convert_targets = None
        else:
            # pascal
            self.convert_targets = np.vectorize(PASCAL_ID_MAPPING.get)

        # we generated cityscapes images with max margin earlier
        self.img_dir = os.path.join(data_path, f'img_with_margin_{self.image_margin_size}/{split_key}')

        if push_prototypes:
            transform = None
        elif 'isbi' in data_path and split_key == 'train':
            transform = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            ])
        else:
            transform = transforms.Compose([
                transforms.Normalize(mean, std)
            ])

        if split_key == 'train' and 'isbi' in data_path:
            self.affine_transform = transforms.RandomAffine(0.2, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=0.05)
        else:
            self.affine_transform = None

        super(PatchClassificationDataset, self).__init__(
            root=self.img_dir,
            transform=transform
        )

        with open(os.path.join(data_path, 'all_images.json'), 'r') as fp:
            self.img_ids = json.load(fp)[split_key]

        self.img_id2idx = {img_id: i for i, img_id in enumerate(self.img_ids)}

        log(f"Loaded {len(self.img_ids)} samples from {split_key} set")

    def __len__(self) -> int:
        if 'isbi' in data_path and not self.push_prototypes:
            return ISBI_MULTIPLIER * len(self.img_ids)
        return len(self.img_ids)

    def get_img_path(self, img_id: str) -> str:
        return os.path.join(self.img_dir, img_id + '.png')

    def __getitem__(self, index: int) -> Any:
        if 'isbi' in data_path and not self.push_prototypes:
            index = index // ISBI_MULTIPLIER

        img_id = self.img_ids[index]
        img_path = os.path.join(self.img_dir, img_id + '.npy')
        label_path = os.path.join(self.annotations_dir, img_id + '.npy')

        image = np.load(img_path).astype(np.uint8)
        label = np.load(label_path)

        if label.ndim == 3:
            label = label[:, :, 0]

        if self.convert_targets is not None:
            label = self.convert_targets(label)
        label = label.astype(np.int32)

        if self.image_margin_size != 0:
            image = image[self.image_margin_size:-self.image_margin_size,
                          self.image_margin_size:-self.image_margin_size]

        h, w = label.shape

        if len(self.scales) < 2:
            scale_factor = 1.0
        else:
            scale_factor = random.uniform(self.scales[0], self.scales[1])
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # [0-255] to [0-1]
        image = image / 255.0

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.window_size[0] - h, 0)
        pad_w = max(self.window_size[1] - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.window_size[0])
        start_w = random.randint(0, w - self.window_size[1])
        end_h = start_h + self.window_size[0]
        end_w = start_w + self.window_size[1]
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()  # HWC
            label = np.fliplr(label).copy()  # HW

        image = torch.tensor(image)
        # HWC -> CHW
        image = image.permute(2, 0, 1)
        label = torch.tensor(label)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # apply affine transforms to both image and labels
        if self.affine_transform is not None:
            concat_img = torch.cat((image, label.unsqueeze(0)), dim=0)
            image = self.affine_transform(concat_img)

            label = torch.round(image[3])
            image = image[:3]

        return image, label

