"""
Pytorch Lightning Module for training prototype image classification model
"""
import os
from typing import Dict

import gin
import torch
from pytorch_lightning import LightningModule
import numpy as np

from helpers import list_of_distances
from model import PPNet
from save import save_model_w_condition
from settings import log
from train_and_test import warm_only, joint, last_only


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reset_metrics() -> Dict:
    return {
        'n_examples': 0,
        'n_correct': 0,
        'n_batches': 0,
        'cross_entropy': 0,
        'cluster_cost': 0,
        'separation': 0,
        'loss': 0,
        'border_cluster_cost': 0
    }


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_dir', 'model_image_size', 'ppnet', 'last_layer_only'])
class ImageClassificationModule(LightningModule):
    def __init__(
            self,
            model_dir: str,
            model_image_size: int,
            ppnet: PPNet,
            last_layer_only: bool,
            num_warm_epochs: int = gin.REQUIRED,
            loss_weight_crs_ent: float = gin.REQUIRED,
            loss_weight_clst: float = gin.REQUIRED,
            loss_weight_sep: float = gin.REQUIRED,
            loss_weight_l1: float = gin.REQUIRED,
            loss_weight_border: float = gin.REQUIRED,
            joint_optimizer_lr_features: float = gin.REQUIRED,
            joint_optimizer_lr_add_on_layers: float = gin.REQUIRED,
            joint_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
            joint_optimizer_weight_decay: float = gin.REQUIRED,
            warm_optimizer_lr_add_on_layers: float = gin.REQUIRED,
            warm_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
            warm_optimizer_weight_decay: float = gin.REQUIRED,
            last_layer_optimizer_lr: float = gin.REQUIRED,
            lr_step_size: int = gin.REQUIRED,
            lr_gamma: float = gin.REQUIRED,
            gradient_clipping: float = gin.REQUIRED
    ):
        super().__init__()
        self.model_dir = model_dir
        self.prototypes_dir = os.path.join(model_dir, 'prototypes')
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.model_image_size = model_image_size
        self.ppnet = ppnet
        self.num_warm_epochs = num_warm_epochs
        self.last_layer_only = last_layer_only
        self.loss_weight_crs_ent = loss_weight_crs_ent
        self.loss_weight_clst = loss_weight_clst
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_l1 = loss_weight_l1
        self.loss_weight_border = loss_weight_border
        self.joint_optimizer_lr_features = joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_weight_decay = joint_optimizer_weight_decay
        self.warm_optimizer_lr_add_on_layers = warm_optimizer_lr_add_on_layers
        self.warm_optimizer_lr_prototype_vectors = warm_optimizer_lr_prototype_vectors
        self.warm_optimizer_weight_decay = warm_optimizer_weight_decay
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.gradient_clipping = gradient_clipping

        os.makedirs(self.prototypes_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize variables for computing metrics
        self.metrics = {}
        for split_key in ['train', 'val', 'test']:
            self.metrics[split_key] = reset_metrics()

        # initialize configure_optimizers()
        self.lr_scheduler = None

        # we use optimizers manually
        self.automatic_optimization = False

        self.best_acc = 0.0

    def save_batch_info(self, image, target, batch_key):
        image = image.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        debug_dir = os.path.join(self.model_dir, 'debug_batches')
        os.makedirs(debug_dir, exist_ok=True)
        np.savez(os.path.join(debug_dir, batch_key), image=image, target=target)

    def forward(self, x):
        return self.ppnet(x)

    def _step(self, split_key: str, batch, batch_idx):
        metrics = self.metrics[split_key]

        image, target = batch

        image = image.to(self.device)
        target = target.to(self.device).to(torch.float32)

        output, min_distances, patch_distances = self.ppnet.forward(image, return_distances=True )
        # compute loss
        cross_entropy = torch.nn.functional.cross_entropy(output, target.long())

        max_dist = (self.ppnet.prototype_shape[1]
                    * self.ppnet.prototype_shape[2]
                    * self.ppnet.prototype_shape[3])

        # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
        # calculate cluster cost
        prototypes_of_correct_class = torch.t(torch.index_select(
            self.ppnet.prototype_class_identity.to(self.device),
            dim=-1,
            index=target.long()
        )).to(self.device)
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
        separation = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        # calculate 'border cluster cost'
        # each image should have some prototype at its borders
        prototype_of_patch_class = prototypes_of_correct_class.unsqueeze(-1)
        min_dist_border1, _ = torch.max((max_dist - patch_distances[:, :, 0, :]) * prototype_of_patch_class, dim=-1)
        min_dist_border2, _ = torch.max((max_dist - patch_distances[:, :, :, 0]) * prototype_of_patch_class, dim=-1)
        min_dist_border3, _ = torch.max((max_dist - patch_distances[:, :, -1, :]) * prototype_of_patch_class, dim=-1)
        min_dist_border4, _ = torch.max((max_dist - patch_distances[:, :, :, -1]) * prototype_of_patch_class, dim=-1)

        max_border_dist = torch.maximum(max_dist - min_dist_border1, max_dist - min_dist_border2)
        max_border_dist = torch.maximum(max_border_dist, max_dist - min_dist_border3)
        max_border_dist = torch.maximum(max_border_dist, max_dist - min_dist_border4)

        border_cluster_cost = torch.mean(max_border_dist)

        l1_mask = 1 - torch.t(self.ppnet.prototype_class_identity).to(self.device)
        l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

        # loss is sometimes Nan or Inf, we need to debug this
        loss = 0
        for key, weight, val in [('cross_entropy', self.loss_weight_crs_ent, cross_entropy),
                                 ('cluster_cost', self.loss_weight_clst, cluster_cost),
                                 ('separation', self.loss_weight_sep, separation),
                                 ('border_cluster_cost', self.loss_weight_border, border_cluster_cost)]:
            if torch.isnan(val):
                log(f'{key} - NaN val')
                self.save_batch_info(image, target, f'nan_{key}_{self.trainer.global_step}')
            elif torch.isinf(val):
                log(f'{key} - inf val')
                self.save_batch_info(image, target, f'inf_{key}_{self.trainer.global_step}')
            else:
                loss += weight * val
                metrics[key] += val.item()

        if self.last_layer_only:
            loss += self.loss_weight_l1 * l1

        loss_value = loss.item() if torch.is_tensor(loss) else 0.0

        metrics['loss'] += loss_value
        metrics['n_examples'] += target.shape[0]
        _, predicted = torch.max(output.data, 1)
        n_correct = (predicted == target).sum().item()
        metrics['n_correct'] += n_correct
        metrics['n_batches'] += 1

        if split_key == 'train':
            warm_optim, main_optim = self.optimizers()

            if self.last_layer_only:
                optimizer = main_optim
            else:
                if self.current_epoch < self.num_warm_epochs:
                    optimizer = warm_optim
                else:
                    optimizer = main_optim

            if torch.is_tensor(loss):
                optimizer.zero_grad()
                self.manual_backward(loss)

                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.ppnet.parameters(), self.gradient_clipping)

                optimizer.step()

            self.log('train_loss_step', loss_value, on_step=True, prog_bar=True)
            self.log('train_accuracy_step', n_correct / target.shape[0], on_step=True, prog_bar=True)

            lr = get_lr(optimizer)
            self.log('lr', lr, on_step=True)

    def training_step(self, batch, batch_idx):
        return self._step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step('test', batch, batch_idx)

    def on_train_epoch_start(self):
        if self.last_layer_only:
            last_only(model=self.ppnet, log=log)
        else:
            if self.current_epoch == 0:
                log('WARM-UP START.')

            if self.current_epoch < self.num_warm_epochs:
                warm_only(model=self.ppnet, log=log)
            else:
                joint(model=self.ppnet, log=log)
                if self.current_epoch == self.num_warm_epochs:
                    log('WARM-UP END.')

        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()

    def on_validation_epoch_end(self):
        val_acc = self.metrics['val']['n_correct'] / self.metrics['val']['n_examples']

        if self.last_layer_only:
            self.log('training_stage', 2.0)
            stage_key = 'push'
        else:
            if self.current_epoch < self.num_warm_epochs:
                # noinspection PyUnresolvedReferences
                self.log('training_stage', 0.0)
                stage_key = 'warmup'
            else:
                # noinspection PyUnresolvedReferences
                self.log('training_stage', 1.0)
                stage_key = 'nopush'
                self.lr_scheduler.step()

        if val_acc > self.best_acc:
            log(f'Saving best model, accuracy: ' + str(val_acc))
            self.best_acc = val_acc
            save_model_w_condition(
                model=self.ppnet,
                model_dir=self.checkpoints_dir,
                model_name=f'{stage_key}_best',
                accu=val_acc,
                target_accu=0.0,
                log=log
            )
        save_model_w_condition(
            model=self.ppnet,
            model_dir=self.checkpoints_dir,
            model_name=f'{stage_key}_last',
            accu=val_acc,
            target_accu=0.0,
            log=log
        )

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        n_batches = metrics['n_batches']

        for key in ['loss', 'cross_entropy', 'cluster_cost', 'separation', 'border_cluster_cost']:
            self.log(f'{split_key}/{key}', metrics[key] / n_batches)

        self.log(f'{split_key}/accuracy', metrics['n_correct'] / metrics['n_examples'])
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
        warm_optimizer_specs = \
            [
                {
                    'params': self.ppnet.add_on_layers.parameters(),
                    'lr': self.warm_optimizer_lr_add_on_layers,
                    'weight_decay': self.warm_optimizer_weight_decay
                },
                {
                    'params': self.ppnet.prototype_vectors,
                    'lr': self.warm_optimizer_lr_prototype_vectors
                }
            ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        if self.last_layer_only:
            main_optimizer_specs = [
                {
                    'params': self.ppnet.last_layer.parameters(),
                    'lr': self.last_layer_optimizer_lr
                }
            ]
        else:
            main_optimizer_specs = \
                [
                    {
                        'params': self.ppnet.features.parameters(),
                        'lr': self.joint_optimizer_lr_features,
                        'weight_decay': self.joint_optimizer_weight_decay
                    },
                    # bias are now also being regularized
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
        main_optimizer = torch.optim.Adam(main_optimizer_specs)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(main_optimizer,
                                                            step_size=self.lr_step_size, gamma=self.lr_gamma)

        return warm_optimizer, main_optimizer
