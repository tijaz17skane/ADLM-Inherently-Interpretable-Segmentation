"""
Pytorch Lightning DataModule for training prototype segmentation model on Cityscapes and SUN datasets
"""
import multiprocessing
import os

import gin
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from segmentation.dataset import PatchClassificationDataset
from settings import data_path

# Try this out in case of high RAM usage:
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

DataLoader = gin.external_configurable(DataLoader)


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_image_size'])
class PatchClassificationDataModule(LightningDataModule):
    def __init__(
            self,
            model_image_size: int,
            dataloader_n_jobs: int = gin.REQUIRED,
    ):
        super().__init__()
        self.dataloader_n_jobs = dataloader_n_jobs if dataloader_n_jobs != -1 else multiprocessing.cpu_count()
        self.model_image_size = model_image_size

    def prepare_data(self):
        if not os.path.exists(os.path.join(data_path, 'annotations')):
            raise ValueError("Please download dataset and preprocess it using 'preprocess.py' script")

    def get_data_loader(self, dataset: PatchClassificationDataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            shuffle=not dataset.is_eval,
            num_workers=self.dataloader_n_jobs,
        )

    def train_dataloader(self):
        train_split = PatchClassificationDataset(
            split_key='train',
            is_eval=False,
            model_image_size=self.model_image_size
        )
        return self.get_data_loader(train_split)

    def val_dataloader(self):
        val_split = PatchClassificationDataset(
            split_key='val',
            is_eval=True,
            model_image_size=self.model_image_size
        )
        return self.get_data_loader(val_split)

    def test_dataloader(self):
        test_split = PatchClassificationDataset(
            split_key='val',  # We do not have test set for cityscapes
            is_eval=True,
            model_image_size=self.model_image_size
        )
        return self.get_data_loader(test_split)

    def train_push_dataloader(self):
        train_split = PatchClassificationDataset(
            split_key='train',
            is_eval=True,
            model_image_size=self.model_image_size,
            push_prototypes=True
        )
        return self.get_data_loader(train_split)
