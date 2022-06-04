from torchvision import transforms
import torch
import numpy as np
from torchvision import transforms
from segmentation.constants import CITYSCAPES_MEAN, CITYSCAPES_STD

to_tensor = transforms.ToTensor()


to_normalized_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)
])


def get_conv_features(dataset, ppnet, img, window_size, window_shift, batch_size):
    current_batch_img = []
    current_batch_ij = []

    n_row_windows = int(np.ceil((img.width - window_size[0]) / window_shift)) + 1
    n_col_windows = int(np.ceil((img.height - window_size[1]) / window_shift)) + 1

    conv_features = torch.zeros((img.width, img.height, ppnet.prototype_shape[1])).cuda()
    n_preds = torch.zeros((img.width, img.height, 1)).cuda()

    for i in range(n_row_windows):
        for j in range(n_col_windows):
            left = window_shift * i
            right = window_shift * i + window_size[0]
            top = window_shift * j
            bottom = window_shift * j + window_size[1]

            if bottom > img.height:
                top = img.height - window_size[0]
                bottom = img.height

            if right > img.width:
                left = img.width - window_size[1]
                right = img.width

            window = img.crop((left, top, right, bottom))
            window = to_tensor(window)
            if hasattr(dataset, 'transform') and dataset.transform is not None:
                window = dataset.transform(window)
            window = window.cuda()

            current_batch_img.append(window)
            current_batch_ij.append((left, top))

            if len(current_batch_img) >= batch_size:
                current_batch_img = torch.stack(current_batch_img, dim=0)
                features = ppnet.conv_features(current_batch_img)

                for (img_i, img_j), feat in zip(current_batch_ij, features):
                    conv_features[img_i:img_i + window_size[0],
                    img_j:img_j + window_size[1]] += feat.T
                    n_preds[img_i:img_i + window_size[0],
                    img_j:img_j + window_size[1]] += 1

                current_batch_img = []
                current_batch_ij = []

    if len(current_batch_img) > 0:
        current_batch_img = torch.stack(current_batch_img, dim=0)
        features = ppnet.conv_features(current_batch_img)

        for (img_i, img_j), feat in zip(current_batch_ij, features):
            conv_features[img_i:img_i + window_size[0],
            img_j:img_j + window_size[1]] += feat.T
            n_preds[img_i:img_i + window_size[0],
            img_j:img_j + window_size[1]] += 1

    conv_features = conv_features / n_preds
    conv_features = conv_features.permute(2, 0, 1)

    return conv_features


def get_prediction_from_features(ppnet, img, conv_features, window_size):
    img_logits = torch.zeros((img.width, img.height, ppnet.num_classes)).cuda()
    img_distances = torch.zeros((img.width, img.height, ppnet.num_prototypes)).cuda()

    n_row_windows = int(np.ceil(img.width / window_size[0]))
    n_col_windows = int(np.ceil(img.height / window_size[1]))

    for i in range(n_row_windows):
        for j in range(n_col_windows):
            left = window_size[0] * i
            right = window_size[0] * i + window_size[0]
            top = window_size[1] * j
            bottom = window_size[1] * j + window_size[1]

            if bottom > img.height:
                top = img.height - window_size[0]
                bottom = img.height

            if right > img.width:
                left = img.width - window_size[1]
                right = img.width

            window_features = conv_features[:, left:right, top:bottom]

            logits, distances = ppnet.forward_from_conv_features(window_features.unsqueeze(0))

            img_logits[left:right, top:bottom] = logits[0]
            img_distances[left:right, top:bottom] = distances[0].permute(1, 2, 0)

    return img_logits, img_distances


def get_image_segmentation(dataset, ppnet, img, window_size, window_shift, batch_size):
    assert window_shift <= window_size[0]
    assert window_shift <= window_size[1]

    with torch.no_grad():
        conv_features = get_conv_features(dataset, ppnet, img, window_size, window_shift, batch_size)
        img_logits, img_distances = get_prediction_from_features(ppnet, img, conv_features, window_size)

    return {
        'logits': img_logits,
        'distances': img_distances,
        'conv_features': conv_features
    }
