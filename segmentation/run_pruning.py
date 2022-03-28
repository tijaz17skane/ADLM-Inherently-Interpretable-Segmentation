import os

import argh
import gin
import torch
import torch.utils.data

import prune
import save
import train_and_test as tnt
from preprocess import preprocess
from segmentation.data_module import SlidingWindowDataModule
from log import create_logger


def run_pruning(experiment_name: str, k: int = 6, prune_threshold: int = 3, optimize_last_layer: bool = False):
    gin.parse_config_file(os.path.join(os.environ['RESULTS_DIR'], experiment_name, 'config.gin'),
                          skip_unknown=True)
    gin.parse_config('DataLoader.batch_size=256')

    model_path = os.path.join(os.environ['RESULTS_DIR'], experiment_name, 'checkpoints/push_best.pth')
    output_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name, 'pruned')

    os.makedirs(output_dir, exist_ok=True)

    log, logclose = create_logger(log_filename=os.path.join(output_dir, 'prune.log'))

    ppnet = torch.load(model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # load the data
    # TODO use configurable value for model_image_size here
    data_module = SlidingWindowDataModule(model_image_size=224)

    test_loader = data_module.val_dataloader()

    # push set: needed for pruning because it is unnormalized
    train_push_loader = data_module.train_push_dataloader()

    def preprocess_push_input(x):
        return preprocess(x, mean=train_push_loader.dataset.mean, std=train_push_loader.dataset.std)

    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))

    # prune prototypes
    log('prune')
    with torch.no_grad():
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        log(f"Accuracy before pruning: {accu}")

        prune.prune_prototypes(dataloader=train_push_loader,
                               prototype_network_parallel=ppnet_multi,
                               k=k,
                               prune_threshold=prune_threshold,
                               preprocess_input_function=preprocess_push_input,  # normalize
                               original_model_dir=output_dir,
                               epoch_number=0,
                               # model_name=None,
                               log=log,
                               copy_prototype_imgs=False, )
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        log(f"Accuracy after pruning: {accu}")

    save.save_model_w_condition(model=ppnet, model_dir=output_dir,
                                model_name='pruned',
                                accu=accu,
                                target_accu=0.0, log=log)

    # last layer optimization
    if optimize_last_layer:
        # TODO run this in our training module
        train_loader = data_module.train_dataloader()
        train_loader.dataset.length_multiplier = 1
        log('training set size: {0}'.format(len(train_loader.dataset)))
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        coefs = {
            'crs_ent': 1,
            'clst': 0.8,
            'sep': -0.08,
            'l1': 1e-4,
        }

        log('optimize last layer')
        tnt.last_only(model=ppnet_multi, log=log)
        for i in range(100):
            log('iteration: \t{0}'.format(i))
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                        model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                        accu=accu,
                                        target_accu=0.70, log=log)

    logclose()


if __name__ == '__main__':
    argh.dispatch_command(run_pruning)
