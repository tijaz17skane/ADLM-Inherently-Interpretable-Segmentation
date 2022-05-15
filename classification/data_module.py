"""
Pytorch Lightning DataModule for training prototype classification model
"""
import multiprocessing
import os
from typing import Tuple

import gin
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from settings import data_path, log


# Try this out in case of high RAM usage:
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_image_size'])
class ImageClassificationDataModule(LightningDataModule):
    def __init__(
            self,
            model_image_size: int,
            dataloader_n_jobs: int = gin.REQUIRED,
            norm_mean: Tuple[float, float, float] = gin.REQUIRED,
            norm_std: Tuple[float, float, float] = gin.REQUIRED,
            train_batch_size: int = gin.REQUIRED,
            test_batch_size: int = gin.REQUIRED,
            train_push_batch_size: int = gin.REQUIRED,

    ):
        super().__init__()
        self.dataloader_n_jobs = dataloader_n_jobs if dataloader_n_jobs != -1 else multiprocessing.cpu_count()
        self.model_image_size = model_image_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_push_batch_size = train_push_batch_size

    def prepare_data(self):
        if not os.path.exists(data_path):
            raise ValueError("Please download dataset and preprocess it using 'preprocess.py' script")

    def get_data_loader(self, dataset: ImageFolder, shuffle: bool, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            num_workers=self.dataloader_n_jobs,
            **kwargs
        )

    def train_dataloader(self, **kwargs):
        log('Loading train data')
        train_split = ImageFolder(
            os.path.join(data_path, 'train'),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=45, shear=10, scale=(0.5, 1.0)),
                transforms.Resize(size=(self.model_image_size, self.model_image_size)),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]))
        log('Train data loaded!')
        return self.get_data_loader(train_split, shuffle=True, batch_size=self.train_batch_size, **kwargs)

    def val_dataloader(self, **kwargs):
        log('Loading val data')
        val_split = ImageFolder(
            os.path.join(data_path, 'val'),
            transforms.Compose([
                transforms.Resize(size=(self.model_image_size, self.model_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ]))
        log('Val data loaded!')
        return self.get_data_loader(val_split, shuffle=False, batch_size=self.test_batch_size, **kwargs)

    def test_dataloader(self, **kwargs):
        return self.val_dataloader(**kwargs)

    def train_push_dataloader(self, **kwargs):
        log('Loading train (push) data')
        train_push_split = ImageFolder(
            os.path.join(data_path, 'train'),
            transforms.Compose([
                transforms.Resize(size=(self.model_image_size, self.model_image_size)),
                transforms.ToTensor(),
            ]))
        log('Train data loaded!')
        return self.get_data_loader(train_push_split, shuffle=False, batch_size=self.train_push_batch_size, **kwargs)
