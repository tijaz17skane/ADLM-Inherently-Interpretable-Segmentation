import os
import torch
import gin
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
# noinspection PyUnresolvedReferences
from classification.train import train_cls
from settings import data_path
from classification.data_module import ImageClassificationDataModule
from torchvision import transforms
from tqdm import tqdm
import cv2
from helpers import find_high_activation_crop
from matplotlib import cm
import json
import sys


model_path = sys.argv[1]

print(f'Loading model from {model_path}')
config_path = os.path.join(model_path, 'config.gin')
gin.parse_config_file(config_path)

checkpoint_path = os.path.join(model_path, 'checkpoints/push_best.pth')
ppnet = torch.load(checkpoint_path)
ppnet = ppnet.cuda()
ppnet.eval()

data_module = ImageClassificationDataModule(model_image_size=224)

resize = transforms.Compose([
    transforms.Resize(size=(data_module.model_image_size, data_module.model_image_size)),
    transforms.ToTensor(),
])

normalize = transforms.Normalize(mean=data_module.norm_mean, std=data_module.norm_std)


cmap = cm.get_cmap('jet')


for split_key in ['train', 'val']:
    print('SPLIT', split_key)
    img_dir = os.path.join(data_path, split_key)
    output_dir = os.path.join(model_path, 'proto_bounding_boxes', split_key)
    os.makedirs(output_dir, exist_ok=True)

    ALL_CLASSES = list(sorted(os.listdir(img_dir)))
    CLASS2ID = {k: i for i, k in enumerate(ALL_CLASSES)}

    for cat_dir in tqdm(ALL_CLASSES, desc=split_key):
        print("CLASS", cat_dir)
        class_id = CLASS2ID[cat_dir]

        output_cat_dir = os.path.join(output_dir, cat_dir)
        os.makedirs(output_cat_dir, exist_ok=True)

        full_cat_dir = os.path.join(img_dir, cat_dir)
        for file in tqdm(os.listdir(full_cat_dir), desc=cat_dir):
            full_file_path = os.path.join(full_cat_dir, file)

            with open(full_file_path, 'rb') as f:
                img = Image.open(f).convert('RGB')

            resized_img = resize(img).cuda()
            normalized_img = normalize(resized_img)

            conv_output, distances = ppnet.push_forward(normalized_img.unsqueeze(0))

            correct_proto_distances = distances[:, class_id * 10 : (class_id + 1) * 10]
            original_img = np.float32(np.asarray(img)) / 255

            proto_activation = torch.log((correct_proto_distances + 1) / (correct_proto_distances + ppnet.epsilon)).cpu().detach().numpy()

            all_proto_bounds = {}

            for img_i in range(proto_activation.shape[0]):
                for proto_i in range(proto_activation.shape[1]):
                    proto_act_full = cv2.resize(proto_activation[img_i, proto_i], dsize=(img.size[0], img.size[1]), 
                                                interpolation=cv2.INTER_CUBIC)

                    # heatmap = cv2.applyColorMap(np.uint8(255*proto_act_full), cv2.COLORMAP_JET)
                    # heatmap = np.float32(heatmap) / 255
                    # heatmap = heatmap[...,::-1]
                    # overlayed_img = 0.5 * original_img + 0.3 * heatmap

                    proto_bounds = find_high_activation_crop(proto_act_full)
                    proto_img = original_img[proto_bounds[0]:proto_bounds[1], proto_bounds[2]:proto_bounds[3], :]
                    all_proto_bounds[10 * class_id + proto_i] = proto_bounds

            plt.figure()
            plt.axis('off')
            plt.imshow(original_img)

            for i, proto_bounds in enumerate(all_proto_bounds.values()):
                plt.plot([proto_bounds[2], proto_bounds[2]], [proto_bounds[0], proto_bounds[1]],
                         [proto_bounds[3], proto_bounds[3]], [proto_bounds[0], proto_bounds[1]],
                         [proto_bounds[2], proto_bounds[3]], [proto_bounds[0], proto_bounds[0]],
                         [proto_bounds[2], proto_bounds[3]], [proto_bounds[1], proto_bounds[1]],
                         color=cmap(int(i*255/10)), linewidth=2)

            plt.savefig(os.path.join(output_cat_dir, file))
            plt.close()
            with open(os.path.join(output_cat_dir, file.replace('.jpg', '.json')), 'w') as f:
                json.dump(all_proto_bounds, f)
