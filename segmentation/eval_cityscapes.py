import json
import os
from collections import Counter

import argh
import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms

from segmentation.constants import CITYSCAPES_CATEGORIES, CITYSCAPES_19_EVAL_CATEGORIES
from settings import data_path, log


def run_evaluation(model_name: str, training_phase: str, batch_size: int = 4):
    model_path = os.path.join(os.environ['RESULTS_DIR'], model_name)
    log(f'Loading model from {model_path}')
    config_path = os.path.join(model_path, 'config.gin')
    gin.parse_config_file(config_path)
    
    if training_phase == 'pruned':
       checkpoint_path = os.path.join(model_path, 'pruned/pruned.pth')
    else:
       checkpoint_path = os.path.join(model_path, f'checkpoints/{training_phase}_last.pth')

    ppnet = torch.load(checkpoint_path)  # , map_location=torch.device('cpu'))
    ppnet = ppnet.cuda()
    ppnet.eval()

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    img_dir = os.path.join(data_path, 'img_with_margin_512/val')
    all_img_files = [p for p in os.listdir(img_dir) if p.endswith('.npy')]

    ann_dir = os.path.join(data_path, 'annotations/val')

    pred2name = {k - 1: i for i, k in CITYSCAPES_19_EVAL_CATEGORIES.items() if k > 0}
    pred2name = {i: CITYSCAPES_CATEGORIES[k] for i, k in pred2name.items()}

    cls_prototype_counts = [Counter() for _ in range(len(pred2name))]
    mean_top_k = np.zeros(200, dtype=float)

    MARGIN = 512
    RESULTS_DIR = os.path.join(model_path, f'evaluation/notebook_plots/{training_phase}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    CLS_CONVERT = np.vectorize(CITYSCAPES_19_EVAL_CATEGORIES.get)

    proto2cls = {}
    cls2protos = {c: [] for c in range(ppnet.num_classes)}
    proto_ident = ppnet.prototype_class_identity.cpu().detach().numpy()

    for proto_num in range(proto_ident.shape[0]):
        cls = np.argmax(proto_ident[proto_num])
        proto2cls[proto_num] = cls
        cls2protos[cls].append(proto_num)

    PROTO2CLS = np.vectorize(proto2cls.get)

    protos = ppnet.prototype_vectors.squeeze()
    n_per_class = 10
    n_classes = protos.shape[0] // n_per_class

    all_cls_distances = []

    with torch.no_grad():
        for cls_i in range(n_classes):
            cls_proto_ind = (proto_ident[:, cls_i] == 1).nonzero()[0]
            cls_protos = protos[cls_proto_ind]

            distances = torch.cdist(cls_protos, cls_protos)
            distances = distances + 10e6 * torch.triu(torch.ones_like(distances, device=cls_protos.device))
            distances = distances.flatten()
            distances = distances[distances < 10e6]

            distances = distances.cpu().detach().numpy()
            all_cls_distances.append(distances)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    plt.suptitle(f'{model_name} ({training_phase})\nHistogram of distances between same-class prototypes')
    axes = axes.flatten()
    for class_i, class_name in pred2name.items():
        axes[class_i].hist(all_cls_distances[class_i], bins=10)
        d_min, d_avg, d_max = np.min(all_cls_distances[class_i]), np.mean(all_cls_distances[class_i]), np.max(
            all_cls_distances[class_i])
        axes[class_i].set_title(f'{class_name}\nmin: {d_min:.2f} avg: {d_avg:.2f} max: {d_max:.2f}')

    axes[-1].axis('off')  # there are 19 classes in cityscapes

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'histogram_dist_same_class_prototypes.png'))

    CLS_I = Counter()
    CLS_U = Counter()

    np.random.shuffle(all_img_files)

    n_batches = int(np.ceil(len(all_img_files) / batch_size))
    batched_img_files = np.array_split(all_img_files, n_batches)

    correct_pixels, total_pixels = 0, 0

    with torch.no_grad():
        for batch_img_files in batched_img_files:
            img_tensors = []
            anns = []

            for img_file in batch_img_files:
                img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)
                ann = np.load(os.path.join(ann_dir, img_file))
                ann = CLS_CONVERT(ann)

                img = img[MARGIN:-MARGIN, MARGIN:-MARGIN]
                img_shape = (img.shape[0], img.shape[1])
                img_tensors.append(transform(img))
                anns.append(ann)

            anns = np.stack(anns, axis=0)
            img_tensors = torch.stack(img_tensors, dim=0).cuda()
            logits, distances = ppnet.forward(img_tensors)

            logits = logits.permute(0, 3, 1, 2)

            logits = F.interpolate(logits, size=img_shape, mode='bilinear', align_corners=False)
            distances = F.interpolate(distances, size=img_shape, mode='bilinear', align_corners=False)

            pred = torch.argmax(logits, axis=1).cpu().detach().numpy()
            nearest_proto = torch.argmin(distances, axis=1).cpu().detach().numpy()

            distances = distances.cpu().detach().numpy()

            correct_pixels += np.sum(((pred + 1) == anns) & (anns != 0))
            total_pixels += np.sum(anns != 0)

            for cls_i in range(19):
                pr = pred == cls_i
                gt = anns == cls_i + 1

                CLS_I[cls_i] += np.sum(pr & gt)
                CLS_U[cls_i] += np.sum((pr | gt) & (anns != 0))  # ignore pixels where ground truth is void

            # save some RAM
            del logits, img_tensors

            # calculate statistics of prototypes occurrences as nearest
            nearest_proto_cls = PROTO2CLS(nearest_proto)

            for class_i, class_name in pred2name.items():
                is_class_proto = (pred == class_i) & (nearest_proto_cls == class_i)
                for proto_i, proto_num in enumerate(cls2protos[class_i]):
                    cls_prototype_counts[class_i][proto_i] += np.sum(is_class_proto & (nearest_proto == proto_num))
            del is_class_proto

            # calculate top K nearest prototypes for random sample of pixels for speed
            for sample_i in range(batch_size):
                n_random_pixels = 100

                sample_distances = distances[sample_i]

                rows = np.random.randint(sample_distances.shape[1], size=n_random_pixels)
                cols = np.random.randint(sample_distances.shape[2], size=n_random_pixels)

                sample_distances = sample_distances[:, rows, cols]
                sample_preds = pred[sample_i, rows, cols]

                nearest_pixel_protos = np.argsort(sample_distances, axis=0)
                is_class_proto = PROTO2CLS(nearest_pixel_protos) == sample_preds

                for k in range(nearest_pixel_protos.shape[0]):
                    nearest_k = np.sum(is_class_proto[:k + 1]) / (k + 1)
                    mean_top_k[k] += nearest_k * 100 / n_random_pixels

    pixel_accuracy = correct_pixels / total_pixels * 100
    reverse_cat_dict = {v: k for k, v in CITYSCAPES_19_EVAL_CATEGORIES.items()}

    CLS_IOU = {cls_i + 1: (CLS_I[cls_i] * 100) / u for cls_i, u in CLS_U.items() if u > 0}
    mean_iou = np.mean(list(CLS_IOU.values()))
    keys = list(sorted(CLS_IOU.keys()))

    vals = [CLS_IOU[k] for k in keys]
    keys = [CITYSCAPES_CATEGORIES[reverse_cat_dict[cls_i]] for cls_i in keys]

    plt.figure(figsize=(15, 5))
    xticks = np.arange(len(keys))
    plt.bar(xticks, vals)
    plt.xticks(xticks, keys, rotation=45)
    plt.title(
        f'{model_name} ({training_phase})\nIOU scores over all {len(CLS_IOU)} classes (mean IOU: {mean_iou:.4f}, pixel accuracy: {pixel_accuracy:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'iou_scores.png'))

    with open(os.path.join(RESULTS_DIR, 'iou_scores.json'), 'w') as fp:
        json.dump(CLS_IOU, fp)

    plt.figure(figsize=(10, 5))
    plt.title(f'{model_name} ({training_phase})\nHow many of the nearest K prototypes to a random pixel are from its predicted class?')
    plt.xlabel('Nearest K prototypes to a pixel')
    plt.ylabel('% of K prototypes from pixel class')
    plt.ylim([0, 100])
    xticks = np.arange(20) * 10
    plt.xticks(xticks, xticks)
    plt.plot(mean_top_k / (len(batched_img_files) * batch_size))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_prototypes_in_nearest_k.png'))

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    plt.suptitle(f'{model_name} ({training_phase})\nOccurences (%) of 10 prototypes of each class in its top nearest class for each pixel')
    axes = axes.flatten()
    for class_i, class_name in pred2name.items():
        n, c = zip(*cls_prototype_counts[class_i].most_common())
        if sum(cls_prototype_counts[class_i].values()) > 0:
            c = c / sum(cls_prototype_counts[class_i].values()) * 100
        axes[class_i].bar(np.arange(len(c)), c)
        axes[class_i].set_xticks(np.arange(len(c)), n)
        axes[class_i].set_title(class_name)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'occurences_of_each_class_prototypes_in_nearest_pixel.png'))

    # run the following code to visualize on some samples

    N_SAMPLES = 20
    DPI = 100

    for example_i, img_file in enumerate(np.random.choice(all_img_files, size=N_SAMPLES, replace=False)):
        img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)

        ann = np.load(os.path.join(ann_dir, img_file))
        ann = np.vectorize(CITYSCAPES_19_EVAL_CATEGORIES.get)(ann)

        img = img[MARGIN:-MARGIN, MARGIN:-MARGIN]
        img_shape = (img.shape[0], img.shape[1])

        with torch.no_grad():
            img_tensor = transform(img).unsqueeze(0).cuda()
            logits, distances = ppnet.forward(img_tensor)

            logits = logits.permute(0, 3, 1, 2)

            logits = F.interpolate(logits, size=img_shape, mode='bilinear', align_corners=False)[0]
            distances = F.interpolate(distances, size=img_shape, mode='bilinear', align_corners=False)[0]

            # (H, W, C)
            distances = distances.cpu().detach().numpy()
            logits = logits.cpu().detach().numpy()

        # nearest_proto = np.argmin(distances_interp, axis=0).T % 10
        nearest_proto = np.argmin(distances, axis=0) % 10
        pred = np.argmax(logits, axis=0)

        # save some RAM
        del distances, logits, img_tensor

        void_mask = (ann == 0).astype(float)

        plt.figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI))
        plt.title(f'{model_name} ({training_phase})\nExample {example_i}. Prediction (from interpolated logits)')
        plt.imshow(img)
        plt.imshow(pred, alpha=0.5)
        plt.imshow(np.zeros_like(pred), alpha=void_mask, vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'example_{example_i}_prediction.png'))

        # show only one example in notebook
        if example_i == 0:
            plt.show()
        plt.close()

        plt.figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI))
        plt.title(f'{model_name} ({training_phase})\nExample {example_i}. Nearest prototypes (from interpolated distances)')
        plt.imshow(img)
        plt.imshow(nearest_proto, alpha=0.5, vmin=0, vmax=9)
        plt.imshow(np.zeros_like(pred), alpha=void_mask, vmin=0, vmax=1, cmap='gray')
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join(RESULTS_DIR, f'example_{example_i}_prototypes.png'))

        plt.close()


if __name__ == '__main__':
    argh.dispatch_command(run_evaluation)
