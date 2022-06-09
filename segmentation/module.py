"""
Pytorch Lightning Module for training prototype segmentation model on Cityscapes and SUN datasets
"""
import os
from typing import Dict, Optional

import gin
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from helpers import list_of_distances
from model import PPNet
from settings import log
from train_and_test import warm_only, joint, last_only


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reset_metrics() -> Dict:
    return {
        'n_correct': 0,
        'n_batches': 0,
        'n_patches': 0,
        'cross_entropy': 0,
        'loss': 0
    }


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_dir', 'ppnet', 'training_phase', 'max_steps'])
class PatchClassificationModule(LightningModule):
    def __init__(
            self,
            model_dir: str,
            ppnet: PPNet,
            training_phase: int,
            max_steps: Optional[int] = None,
            poly_lr_power: float = gin.REQUIRED,
            loss_weight_crs_ent: float = gin.REQUIRED,
            loss_weight_l1: float = gin.REQUIRED,
            joint_optimizer_lr_features: float = gin.REQUIRED,
            joint_optimizer_lr_add_on_layers: float = gin.REQUIRED,
            joint_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
            joint_optimizer_weight_decay: float = gin.REQUIRED,
            warm_optimizer_lr_add_on_layers: float = gin.REQUIRED,
            warm_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
            warm_optimizer_weight_decay: float = gin.REQUIRED,
            last_layer_optimizer_lr: float = gin.REQUIRED,
            warmup_batches: int = gin.REQUIRED,
            ignore_void_class: bool = False
    ):
        super().__init__()
        self.model_dir = model_dir
        self.prototypes_dir = os.path.join(model_dir, 'prototypes')
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.ppnet = ppnet
        self.training_phase = training_phase
        self.max_steps = max_steps
        self.poly_lr_power = poly_lr_power
        self.loss_weight_crs_ent = loss_weight_crs_ent
        self.loss_weight_l1 = loss_weight_l1
        self.joint_optimizer_lr_features = joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_weight_decay = joint_optimizer_weight_decay
        self.warm_optimizer_lr_add_on_layers = warm_optimizer_lr_add_on_layers
        self.warm_optimizer_lr_prototype_vectors = warm_optimizer_lr_prototype_vectors
        self.warm_optimizer_weight_decay = warm_optimizer_weight_decay
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        self.warmup_batches = warmup_batches
        self.ignore_void_class = ignore_void_class

        os.makedirs(self.prototypes_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize variables for computing metrics
        self.metrics = {}
        for split_key in ['train', 'val', 'test', 'train_last_layer']:
            self.metrics[split_key] = reset_metrics()

        # initialize configure_optimizers()
        self.optimizer_defaults = None
        self.start_step = None

        # we use optimizers manually
        self.automatic_optimization = False
        self.best_acc = 0.0

        if self.training_phase == 0:
            warm_only(model=self.ppnet, log=log)
            log(f'WARM-UP TRAINING START. ({self.max_steps} steps)')
        elif self.training_phase == 1:
            joint(model=self.ppnet, log=log)
            log(f'JOINT TRAINING START. ({self.max_steps} steps)')
        else:
            last_only(model=self.ppnet, log=log)
            log('LAST LAYER TRAINING START.')

    def forward(self, x):
        return self.ppnet(x)

    def _step(self, split_key: str, batch):
        if self.start_step is None:
            self.start_step = self.trainer.global_step

        self.ppnet.features.freeze_bn()

        metrics = self.metrics[split_key]

        image, target = batch

        image = image.to(self.device)
        target = target.to(self.device).to(torch.float32)
        output, patch_distances = self.ppnet.forward(image)
        del patch_distances  # optimize GPU RAM

        # we flatten target/output - classification is done per patch
        output = output.reshape(-1, output.shape[-1])
        target = target.flatten()

        if self.ignore_void_class:
            # do not predict label for void class
            target_not_void = (target != 0).nonzero().squeeze()
            target = target[target_not_void] - 1
            output = output[target_not_void]

        cross_entropy = torch.nn.functional.cross_entropy(
            output,
            target.long(),
        )

        output_sigmoid = torch.softmax(output, dim=-1)
        output_class = torch.argmax(output_sigmoid, dim=-1)
        is_correct = output_class == target

        l1_mask = 1 - torch.t(self.ppnet.prototype_class_identity).to(self.device)
        l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

        loss = self.loss_weight_crs_ent * cross_entropy + self.loss_weight_l1 * l1
        loss_value = loss.item()

        metrics['loss'] += loss_value
        metrics['cross_entropy'] += cross_entropy.item()
        metrics['n_correct'] += torch.sum(is_correct)
        metrics['n_patches'] += output.shape[0]
        metrics['n_batches'] += 1

        if split_key == 'train':
            optimizer = self.optimizers()

            optimizer.zero_grad()
            self.manual_backward(loss)

            optimizer.step()

            self.log('train_loss_step', loss_value, on_step=True, prog_bar=True)

            lr = get_lr(optimizer)
            self.log('lr', lr, on_step=True)

            if self.training_phase != 2:  # LR schedule is not used in last layer fine-tuning
                if self.optimizer_defaults is None:
                    self.optimizer_defaults = []
                    for pg in optimizer.param_groups:
                        self.optimizer_defaults.append(pg['lr'])

                for default, pg in zip(self.optimizer_defaults, optimizer.param_groups):
                    pg['lr'] = ((1 - (self.trainer.global_step - self.start_step) /
                                 self.current_max_steps) ** self.poly_lr_power) * default

    def training_step(self, batch, batch_idx):
        return self._step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self._step('val', batch)

    def test_step(self, batch, batch_idx):
        return self._step('test', batch)

    def on_train_epoch_start(self):
        if self.last_layer_only:
            last_only(model=self.ppnet, log=log)
        else:
            if self.current_epoch < self.num_warm_epochs:
                if self.current_epoch == 0:
                    log('WARM-UP TRAINING START.')
                warm_only(model=self.ppnet, log=log)
            else:
                joint(model=self.ppnet, log=log)
                if self.current_epoch == self.num_warm_epochs:
                    log('JOINT TRAINING START.')

        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()

        # Freeze the pre-trained batch norm
        self.ppnet.features.freeze_bn()

    def on_validation_epoch_end(self):
        val_acc = (self.metrics['val']['n_correct'] / self.metrics['val']['n_patches']).item()

        self.log('training_stage', self.training_phase)

        if self.training_phase == 0:
            stage_key = 'warmup'
        elif self.training_phase == 1:
            stage_key = 'nopush'
        else:
            stage_key = 'push'

        torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f'{stage_key}_last.pth'))

        if val_acc > self.best_acc:
            log(f'Saving best model, accuracy: ' + str(val_acc))
            self.best_acc = val_acc
            torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f'{stage_key}_best.pth'))

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        n_batches = metrics['n_batches']

        for key in ['loss', 'cross_entropy']:
            self.log(f'{split_key}/{key}', metrics[key] / n_batches)

        self.log(f'{split_key}/accuracy', metrics['n_correct'] / metrics['n_patches'])
        self.log('l1', self.ppnet.last_layer.weight.norm(p=1).item())

    def training_epoch_end(self, step_outputs):
        return self._epoch_end('train')

    def validation_epoch_end(self, step_outputs):
        p = self.ppnet.prototype_vectors.view(self.ppnet.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log('p dist pair', p_avg_pair_dist.item())

        return self._epoch_end('val')

    def test_epoch_end(self, step_outputs):
        return self._epoch_end('test')

    def configure_optimizers(self):
        if self.training_phase == 0:  # warmup
            aspp_params = [
                self.ppnet.features.aspp.c0.weight,
                self.ppnet.features.aspp.c0.bias,
                self.ppnet.features.aspp.c1.weight,
                self.ppnet.features.aspp.c1.bias,
                self.ppnet.features.aspp.c2.weight,
                self.ppnet.features.aspp.c2.bias,
                self.ppnet.features.aspp.c3.weight,
                self.ppnet.features.aspp.c3.bias
            ]
            optimizer_specs = \
                [
                    {
                        'params': list(self.ppnet.add_on_layers.parameters()) + aspp_params,
                        'lr': self.warm_optimizer_lr_add_on_layers,
                        'weight_decay': self.warm_optimizer_weight_decay
                    },
                    {
                        'params': self.ppnet.prototype_vectors,
                        'lr': self.warm_optimizer_lr_prototype_vectors
                    }
                ]
        elif self.training_phase == 1:  # joint
            optimizer_specs = \
                [
                    {
                        'params': self.ppnet.features.parameters(),
                        'lr': self.joint_optimizer_lr_features,
                        'weight_decay': self.joint_optimizer_weight_decay
                    },
                    {
                        'params': self.ppnet.add_on_layers.parameters(),
                        'lr': self.joint_optimizer_lr_add_on_layers,
                        'weight_decay': self.joint_optimizer_weight_decay
                    },
                    {
                        'params': self.ppnet.prototype_vectors,
                        'lr': self.joint_optimizer_lr_prototype_vectors
                    }
                ]
        else:  # last layer
            optimizer_specs = [
                {
                    'params': self.ppnet.last_layer.parameters(),
                    'lr': self.last_layer_optimizer_lr
                }
            ]
        return torch.optim.Adam(optimizer_specs)
