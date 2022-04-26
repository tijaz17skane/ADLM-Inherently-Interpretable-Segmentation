"""
Pytorch Lightning Module for training prototype segmentation model on Cityscapes and SUN datasets
"""
import os
from typing import Dict, Optional

import gin
import numpy as np
import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from helpers import list_of_distances
from model import PPNet
from save import save_model_w_condition
from settings import log
from train_and_test import warm_only, joint, last_only

ReduceLROnPlateau = gin.external_configurable(ReduceLROnPlateau)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reset_metrics() -> Dict:
    return {
        'n_examples': 0,
        'n_correct': 0,
        'pos_scores': [],
        'neg_scores': [],
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'n_batches': 0,
        'n_patches': 0,
        'cross_entropy': 0,
        'cluster_cost': 0,
        'separation': 0,
        'separation_higher': 0,
        'contrastive_loss': 0,
        'loss': 0
    }


def update_lr_warmup(optimizer, current_step, warmup_batches):
    if warmup_batches > 0:
        if current_step < warmup_batches:
            lr_scale = min(1., float(current_step + 1) / warmup_batches)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * optimizer.defaults['lr']


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_dir', 'model_image_size', 'ppnet', 'last_layer_only'])
class PatchClassificationModule(LightningModule):
    def __init__(
            self,
            model_dir: str,
            model_image_size: int,
            ppnet: PPNet,
            last_layer_only: bool,
            num_warm_epochs: int = gin.REQUIRED,
            loss_weight_crs_ent: float = gin.REQUIRED,
            loss_weight_contrastive: float = gin.REQUIRED,
            loss_weight_clst: float = gin.REQUIRED,
            loss_weight_sep: float = gin.REQUIRED,
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
            gradient_clipping: Optional[float] = gin.REQUIRED,
            ignore_void_class: bool = False
    ):
        super().__init__()
        self.model_dir = model_dir
        self.prototypes_dir = os.path.join(model_dir, 'prototypes')
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.model_image_size = model_image_size
        self.ppnet = ppnet
        self.last_layer_only = last_layer_only
        self.num_warm_epochs = num_warm_epochs
        self.loss_weight_crs_ent = loss_weight_crs_ent
        self.loss_weight_contrastive = loss_weight_contrastive
        self.loss_weight_clst = loss_weight_clst
        self.loss_weight_sep = loss_weight_sep
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
        self.gradient_clipping = gradient_clipping
        self.ignore_void_class = ignore_void_class

        os.makedirs(self.prototypes_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize variables for computing metrics
        self.metrics = {}
        for split_key in ['train', 'val', 'test', 'train_last_layer']:
            self.metrics[split_key] = reset_metrics()

        # initialize configure_optimizers()
        self.lr_scheduler = None

        # we use optimizers manually
        self.automatic_optimization = False

        self.best_loss = 10e6

        # use this for distribution prediction
        # self.loss = torch.nn.KLDivLoss(reduction='batchmean')

        # self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x):
        return self.ppnet(x)

    def _step(self, split_key: str, batch):
        metrics = self.metrics[split_key]

        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device).to(torch.float32)

        output, patch_distances = self.ppnet.forward(image)

        # treat each patch as a separate sample in calculating loss
        # log_output_flat = torch.nn.functional.log_softmax(output.reshape(-1, output.shape[-1]), dim=-1)
        # target_flat = target.reshape(-1, target.shape[-1])

        output_flat = output.reshape(-1, output.shape[-1])
        dist_flat = patch_distances.permute(0, 2, 3, 1).reshape(-1, patch_distances.shape[1])

        if target.ndim > 3:
            target_flat = target.reshape(-1, target.shape[-1])

            # make positive targets have the same weights as negatives in each patch
            n_neg = torch.sum(1 - target_flat, dim=-1).unsqueeze(-1)
            n_pos = torch.sum(target_flat, dim=-1).unsqueeze(-1)

            pos_weight = n_neg / n_pos
            neg_weight = n_pos / n_neg

            weight = pos_weight * target_flat + neg_weight * (1 - target_flat)

            cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
                output_flat,
                target_flat,
                weight=weight
            )

            output_sigmoid = torch.sigmoid(output_flat)
            output_class = (output_sigmoid > 0.5).to(torch.float)
            is_correct = output_class == target_flat

            target_mask = target_flat.bool()

            metrics['pos_scores'] += list(torch.masked_select(output_sigmoid, target_mask).cpu().detach().numpy())
            metrics['neg_scores'] += list(torch.masked_select(output_sigmoid, ~target_mask).cpu().detach().numpy())

            metrics['tp'] += torch.sum(is_correct * target_flat).item()
            metrics['tn'] += torch.sum(is_correct * (1 - target_flat)).item()
            metrics['fp'] += torch.sum(~is_correct * (1 - target_flat)).item()
            metrics['fn'] += torch.sum(~is_correct * target_flat).item()

            target_oh = target_flat
        else:
            target_flat = target.flatten()

            if self.ignore_void_class:
                target_not_void = (target_flat != 0).nonzero().squeeze()
                target_flat = target_flat[target_not_void] - 1
                output_flat = output_flat[target_not_void]
                dist_flat = dist_flat[target_not_void]

            cross_entropy = torch.nn.functional.cross_entropy(
                output_flat,
                target_flat.long(),
            )

            output_sigmoid = torch.softmax(output_flat, dim=-1)
            output_class = torch.argmax(output_sigmoid, dim=-1)
            is_correct = output_class == target_flat

            target_oh = torch.nn.functional.one_hot(target_flat.long(), num_classes=self.ppnet.num_classes)

        l1_mask = 1 - torch.t(self.ppnet.prototype_class_identity).to(self.device)
        l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

        metrics['n_batches'] += 1

        # calculate cluster and separation losses
        separation, cluster_cost = [], []

        if self.ignore_void_class:
            n_p_per_class = self.ppnet.num_prototypes // self.ppnet.num_classes
        else:
            n_p_per_class = self.ppnet.num_prototypes // (self.ppnet.num_classes - 1)

        # TODO maybe we can do it even smarter without the loop
        for cls_i in range(target_oh.shape[1]):
            if cls_i == 0 and not self.ignore_void_class:
                # ignore 'void' class in loop
                continue

            if self.ignore_void_class:
                cls_dists = dist_flat[:, cls_i * n_p_per_class:(cls_i + 1) * n_p_per_class]
            else:
                cls_dists = dist_flat[:, (cls_i - 1) * n_p_per_class:cls_i * n_p_per_class]

            min_cls_dists, _ = torch.min(cls_dists, dim=-1)
            target_cls = target_oh[:, cls_i]

            # we want to minimize cluster_cost and maximize separation
            separation.append((1 - target_cls) * min_cls_dists + 512 * target_cls)
            cluster_cost.append(min_cls_dists * target_cls)

        # separation cost = minimum over distances to all classes that have score==0.0
        separation = torch.stack(separation, dim=-1)
        separation, _ = torch.min(separation, dim=-1)

        # cluster cost = maximum over distances to all classes that have score!=0.0
        cluster_cost = torch.stack(cluster_cost, dim=-1)
        cluster_cost, _ = torch.max(cluster_cost, dim=-1)

        # try contrastive loss formulation (we want higher 'logits' for separation than for cluster cost)
        contrastive_input = torch.stack((cluster_cost, separation), dim=-1)
        contrastive_target = torch.ones(contrastive_input.shape[0], device=contrastive_input.device, dtype=torch.long)
        contrastive_loss = torch.nn.functional.cross_entropy(contrastive_input, contrastive_target)

        separation_higher = torch.sum(separation > cluster_cost)
        cluster_cost = torch.mean(cluster_cost)
        separation = torch.mean(separation)

        loss = (self.loss_weight_crs_ent * cross_entropy +
                self.loss_weight_contrastive * contrastive_loss +
                self.loss_weight_clst * cluster_cost +
                self.loss_weight_sep * separation +
                self.loss_weight_l1 * l1)

        loss_value = loss.item()

        if not np.isnan(loss_value):
            metrics['loss'] += loss_value
            metrics['cross_entropy'] += cross_entropy.item()
            metrics['contrastive_loss'] += contrastive_loss.item()
            metrics['cluster_cost'] += cluster_cost.item()
            metrics['separation'] += separation.item()
            metrics['separation_higher'] += separation_higher.item()
            metrics['n_examples'] += target_flat.size(0)
            metrics['n_correct'] += torch.sum(is_correct)
            n_patches = output_flat.shape[0]
            metrics['n_patches'] += n_patches

        if split_key == 'train':
            warm_optim, main_optim = self.optimizers()

            if self.last_layer_only:
                optimizer = main_optim
            else:
                if self.current_epoch < self.num_warm_epochs:
                    optimizer = warm_optim
                else:
                    optimizer = main_optim

            update_lr_warmup(optimizer, self.trainer.global_step, self.warmup_batches)

            optimizer.zero_grad()
            self.manual_backward(loss)

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.ppnet.parameters(), self.gradient_clipping)

            optimizer.step()
            self.log('train_loss_step', loss_value, on_step=True, prog_bar=True)

            lr = get_lr(optimizer)
            self.log('lr', lr, on_step=True)

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
        val_loss = self.metrics['val']['cross_entropy'] / self.metrics['val']['n_batches']

        if self.last_layer_only:
            self.log('training_stage', 2.0)
            stage_key = 'push'
            self.lr_scheduler.step(val_loss)
        else:
            if self.current_epoch < self.num_warm_epochs:
                # noinspection PyUnresolvedReferences
                self.log('training_stage', 0.0)
                stage_key = 'warmup'
            else:
                # noinspection PyUnresolvedReferences
                self.log('training_stage', 1.0)
                stage_key = 'nopush'
                self.lr_scheduler.step(val_loss)

        if val_loss < self.best_loss:
            log(f'Saving best model, loss: ' + str(val_loss))
            self.best_loss = val_loss
            save_model_w_condition(
                model=self.ppnet,
                model_dir=self.checkpoints_dir,
                model_name=f'{stage_key}_best',
                accu=val_loss,
                target_accu=0.0,
                log=log
            )
        save_model_w_condition(
            model=self.ppnet,
            model_dir=self.checkpoints_dir,
            model_name=f'{stage_key}_last',
            accu=val_loss,
            target_accu=0.0,
            log=log
        )

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        n_batches = metrics['n_batches']

        for key in ['loss', 'contrastive_loss', 'cross_entropy', 'cluster_cost', 'separation']:
            self.log(f'{split_key}/{key}', metrics[key] / n_batches)

        if len(metrics['pos_scores']) > 0 or len(metrics['neg_scores']) > 0:
            pred_scores = np.concatenate((np.array(metrics['pos_scores']), np.asarray(metrics['neg_scores'])))
            true_scores = np.concatenate((np.ones(len(metrics['pos_scores'])), np.zeros(len(metrics['neg_scores']))))
            true_scores = true_scores.astype(int)

            auroc = roc_auc_score(true_scores, pred_scores)

            self.log(f'{split_key}/auroc', auroc)

            pred_pos = metrics['tp'] + metrics['fp']
            precision = metrics['tp'] / pred_pos if pred_pos != 0 else 1.0
            self.log(f'{split_key}/precision', precision)

            true_pos = metrics['tp'] + metrics['fn']
            recall = metrics['tp'] / true_pos if true_pos != 0 else 1.0
            self.log(f'{split_key}/recall', recall)

            f1 = 2 * (precision * recall) / (precision + recall)
            self.log(f'{split_key}/f1_score', f1)
            self.log(f'{split_key}/accuracy', metrics['n_correct'] / (metrics['n_patches'] * self.ppnet.num_classes))
        else:
            self.log(f'{split_key}/accuracy', metrics['n_correct'] / metrics['n_patches'])

        self.log(f'{split_key}/separation_higher', metrics['separation_higher'] / metrics['n_patches'])
        self.log('l1', self.ppnet.last_layer.weight.norm(p=1).item())

        p = self.ppnet.prototype_vectors.view(self.ppnet.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log('p dist pair', p_avg_pair_dist.item())

    def training_epoch_end(self, step_outputs):
        return self._epoch_end('train')

    def validation_epoch_end(self, step_outputs):
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
        self.lr_scheduler = ReduceLROnPlateau(main_optimizer)

        return warm_optimizer, main_optimizer
