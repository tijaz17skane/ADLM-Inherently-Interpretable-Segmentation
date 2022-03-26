"""
Dataset for training prototype segmentation model on Cityscapes and SUN datasets
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
from PIL import Image

import numpy as np


@gin.configurable(allowlist=['mean', 'std', 'min_window_size', 'max_window_size', 'length_multiplier',
                             'transpose_ann', 'balance_classes'])
class SlidingWindowDataset(VisionDataset):
    def __init__(
            self,
            split_key: str,
            is_eval: bool,
            model_image_size: int,
            push_prototypes: bool = False,
            mean: List[float] = gin.REQUIRED,
            std: List[float] = gin.REQUIRED,
            min_window_size: int = gin.REQUIRED,
            max_window_size: int = gin.REQUIRED,
            length_multiplier: int = gin.REQUIRED,
            transpose_ann: bool = gin.REQUIRED,
            balance_classes: bool = gin.REQUIRED,
    ):
        assert 0 < min_window_size <= max_window_size <= 1024

        if not balance_classes:
            raise NotImplementedError(f'Currently, only balance_classes=True is implemented.')

        self.mean = mean
        self.std = std
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.model_image_size = model_image_size
        self.is_eval = is_eval
        self.split_key = split_key
        self.annotations_dir = os.path.join(data_path, 'annotations', split_key)
        self.push_prototypes = push_prototypes
        self.length_multiplier = length_multiplier
        self.transpose_ann = transpose_ann
        self.balance_classes = balance_classes

        # we generated cityscapes images with max margin of 512 earlier
        self.img_dir = os.path.join(data_path, f'img_with_margin_512/{split_key}')

        if push_prototypes:
            transform = transforms.Compose([
                transforms.Resize((self.model_image_size, self.model_image_size)),
                transforms.ToTensor()
            ])
        elif self.is_eval:
            transform = transforms.Compose([
                transforms.Resize((self.model_image_size, self.model_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.Resize((self.model_image_size, self.model_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        super(SlidingWindowDataset, self).__init__(
            root=self.img_dir,
            transform=transform
        )

        with open(os.path.join(data_path, 'all_images.json'), 'r') as fp:
            self.img_ids = json.load(fp)[split_key]

        self.img_id2idx = {img_id: i for i, img_id in enumerate(self.img_ids)}
        self.cls2images = np.load(os.path.join(data_path, 'class2images', split_key, 'cls2img.npz'))
        self.cls2images = {int(k): v for k, v in self.cls2images.items()}
        self.class_nums = list(sorted(self.cls2images.keys()))

        if bool(int(os.environ['LOAD_IMAGES_RAM'])):
            log(f"Loading {len(self.img_ids)} samples from {split_key} set to memory...")
            self.loaded_images, self.loaded_annotations = self.load_images()
        else:
            self.loaded_images, self.loaded_annotations = None, None

        log(f"Loaded {len(self.img_ids)} samples from {split_key} set")

        self.cached_img = None
        self.cached_img_id = None

    def __len__(self) -> int:
        return int(len(self.cls2images) * self.length_multiplier)

    def get_img_path(self, img_id: str) -> str:
        return os.path.join(self.img_dir, img_id + '.png')

    def load_images(self):
        images = []
        annotations = []

        for img_id in self.img_ids:
            img_path = os.path.join(self.img_dir, img_id + '.png')
            ann_path = os.path.join(self.annotations_dir, img_id + '.npy')

            with Image.open(img_path) as img:
                images.append(img.convert('RGB'))
            ann = np.load(ann_path)
            if self.transpose_ann:
                ann = ann.transpose()
            annotations.append(ann)

        return images, annotations

    def __getitem__(self, index: int) -> Any:
        try:
            target = int(index / self.length_multiplier)
            target = self.class_nums[target]

            img_id = np.random.choice(self.cls2images[target])
            img_index = self.img_id2idx[img_id]

            if self.loaded_images is None:
                if self.cached_img_id == img_id and self.cached_img is not None:
                    img = self.cached_img[0]
                    ann = self.cached_img[1]
                else:
                    self.cached_img, self.cached_img_id = None, None
                    img_path = os.path.join(self.img_dir, img_id + '.png')
                    ann_path = os.path.join(self.annotations_dir, img_id + '.npy')

                    with Image.open(img_path) as img:
                        img = img.convert('RGB')

                    ann = np.load(ann_path)
                    if self.transpose_ann:
                        ann = ann.transpose()

                    self.cached_img = img, ann
                    self.cached_img_id = img_id
            else:
                img = self.loaded_images[img_index]
                ann = self.loaded_annotations[img_index]

            cls_idx = np.argwhere(ann == target)
            pixel_i = np.random.randint(0, cls_idx.shape[0])

            # image has additional margin of 512, so we add it to get the central pixel location in the image
            window_center1 = cls_idx[pixel_i, 0] + 512
            window_center2 = cls_idx[pixel_i, 1] + 512

            window_size = np.random.randint(self.min_window_size, self.max_window_size+1)
            margin_size = int(window_size/2)

            img = img.crop((
                window_center1 - margin_size,
                window_center2 - margin_size,
                window_center1 + margin_size,
                window_center2 + margin_size
            ))

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        except Exception as e:
            log(f"EXCEPTION IN DATASET.__getitem__: {str(e)}")
            img = torch.zeros((3, self.model_image_size, self.model_image_size), dtype=torch.float)
            target = torch.tensor(0, dtype=torch.long)
            return img, target


if __name__ == '__main__':
    dataset = SlidingWindowDataset(
        split_key='train',
        is_eval=True,
        min_window_size=224,
        max_window_size=224,
        model_image_size=True
    )

    for i in tqdm(range(len(dataset)), desc='smoke testing dataset'):
        im, tgt = dataset[i]
