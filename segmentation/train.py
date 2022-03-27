"""
Training prototype segmentation model on Cityscapes or SUN dataset

Example run:

python -m segmentation.train cityscapes 2022_03_26_cityscapes
"""
import os
import shutil

import argh
import torch
from pytorch_lightning import Trainer, seed_everything
import gin
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger

from segmentation.data_module import SlidingWindowDataModule
from segmentation.module import SlidingWindowModule
from segmentation.config import get_operative_config_json
from model import construct_PPNet
from preprocess import preprocess
from push import push_prototypes
from settings import log

Trainer = gin.external_configurable(Trainer)


@gin.configurable(allowlist=['model_image_size', 'random_seed',
                             'early_stopping_patience_main', 'early_stopping_patience_last_layer',
                             'start_checkpoint'])
def train(
        config_path: str,
        experiment_name: str,
        pruned: bool = False,
        model_image_size: int = gin.REQUIRED,
        random_seed: int = gin.REQUIRED,
        early_stopping_patience_main: int = gin.REQUIRED,
        early_stopping_patience_last_layer: int = gin.REQUIRED,
        start_checkpoint: str = '',
        start_epoch: int = 0
):
    if start_epoch != 0 and not pruned:
        raise NotImplementedError(f'start_epoch can be only set when pruned=True')

    seed_everything(random_seed)

    results_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    log(f'Starting experiment in "{results_dir}" from config {config_path}')

    if start_checkpoint:
        log(f'Loading checkpoint from {start_checkpoint}')
        ppnet = torch.load(start_checkpoint)
    else:
        ppnet = construct_PPNet(img_size=model_image_size)

    data_module = SlidingWindowDataModule(
        model_image_size=model_image_size,
    )

    logs_dir = os.path.join(results_dir, 'logs')
    os.makedirs(os.path.join(logs_dir, 'tb'), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'csv'), exist_ok=True)

    tb_logger = TensorBoardLogger(logs_dir, name='tb')
    csv_logger = CSVLogger(logs_dir, name='csv')
    loggers = [tb_logger, csv_logger]

    json_gin_config = get_operative_config_json()

    tb_logger.log_hyperparams(json_gin_config)
    csv_logger.log_hyperparams(json_gin_config)

    if not pruned:
        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            neptune_logger = NeptuneLogger(
                project="mikolajsacha/protobased-research",
                tags=[config_path, 'protopnet'],
                name=experiment_name
            )
            loggers.append(neptune_logger)

            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'segmentation/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config

        shutil.copy(f'segmentation/configs/{config_path}.gin', os.path.join(results_dir, 'config.gin'))

        log('MAIN TRAINING')
        callbacks = [
            EarlyStopping(monitor='val/accuracy', patience=early_stopping_patience_main, mode='max')
        ]

        module = SlidingWindowModule(
            model_dir=results_dir,
            model_image_size=model_image_size,
            ppnet=ppnet,
            last_layer_only=False
        )

        trainer = Trainer(logger=loggers, callbacks=callbacks, checkpoint_callback=None,
                          enable_progress_bar=False)
        trainer.fit(model=module, datamodule=data_module)

        best_checkpoint = os.path.join(results_dir, 'checkpoints', 'nopush_best.pth')
        log(f'Loading best model from {best_checkpoint}')
        ppnet = torch.load(best_checkpoint)
        ppnet = ppnet.cuda()

        log('SAVING PROTOTYPES')
        module.eval()
        torch.set_grad_enabled(False)
        push_dataloader = data_module.train_push_dataloader()

        def preprocess_push_input(x):
            return preprocess(x, mean=push_dataloader.dataset.mean, std=push_dataloader.dataset.std)

        push_prototypes(
            push_dataloader,
            prototype_network_parallel=ppnet,
            class_specific=module.class_specific,
            preprocess_input_function=preprocess_push_input,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=module.prototypes_dir,
            epoch_number=module.current_epoch,
            prototype_img_filename_prefix='prototype-img',
            prototype_self_act_filename_prefix='prototype-self-act',
            proto_bound_boxes_filename_prefix='bb',
            save_prototype_class_identity=True,
            log=log
        )
    else:
        best_checkpoint = os.path.join(results_dir, 'pruned/pruned.pth')
        log(f'Loading pruned model from {best_checkpoint}')
        ppnet = torch.load(best_checkpoint)
        ppnet = ppnet.cuda()

    log('LAST LAYER FINE-TUNING')
    callbacks = [
        EarlyStopping(monitor='val/loss', patience=early_stopping_patience_last_layer, mode='min')
    ]

    module = SlidingWindowModule(
        model_dir=os.path.join(results_dir, 'pruned') if pruned else results_dir,
        model_image_size=model_image_size,
        ppnet=ppnet,
        last_layer_only=True
    )

    current_epoch = trainer.current_epoch
    trainer = Trainer(logger=loggers, callbacks=callbacks, checkpoint_callback=None,
                      enable_progress_bar=False)
    if start_epoch != 0:
        trainer.fit_loop.current_epoch = start_epoch
    else:
        trainer.fit_loop.current_epoch = current_epoch + 1
    trainer.fit(model=module, datamodule=data_module)


def load_config_and_train(
        config_path: str,
        experiment_name: str,
        pruned: bool = False,
        start_epoch: int = 0
):
    gin.parse_config_file(f'segmentation/configs/{config_path}.gin')
    train(config_path, experiment_name, pruned=pruned, start_epoch=start_epoch)


if __name__ == '__main__':
    argh.dispatch_command(load_config_and_train)
