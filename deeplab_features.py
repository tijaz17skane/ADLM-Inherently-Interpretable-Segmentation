from typing import Optional

import gin
import torchvision
from torch import nn

from deeplab_pytorch.libs.models.deeplabv2 import DeepLabV2


class DeeplabV3_features(nn.Module):
    def __init__(self, model, layers, **kwargs):
        super(DeeplabV3_features, self).__init__()
        self.model = model
        self.layers = layers

        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

    def forward(self, *args, **kwargs):
        result = self.model.forward(*args, **kwargs)
        return result['out']

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        raise NotImplemented("TODO")


def deeplabv3_resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Coco
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, aux_loss=False)
    # backbone = torchvision.models.resnet50(pretrained=pretrained)
    # from torchvision.models._utils import IntermediateLayerGetter
    # model.backbone = IntermediateLayerGetter(backbone, return_layers={'layer3': 'aux', 'layer4': 'out'})

    model.classifier._modules = {k: model.classifier._modules[k] for k in list(model.classifier._modules.keys())[:-1]}

    # change output stride from 8 to 16
    # TODO change to 8 when evaluating
    # model.backbone.layer4[0].conv2.stride = (2, 2)  # was (1, 1)
    # model.backbone.layer4[0].conv2.padding = (1, 1)  # was (4, 4)
    # model.backbone.layer4[0].conv2.dilation = (1, 1)  # was (4, 4)

    # model.backbone.layer4[0].downsample[0].stride = (2, 2)  # was (1, 1)

    # model.backbone.layer4[1].conv2.padding = (1, 1)  # was (4, 4)
    # model.backbone.layer4[1].conv2.dilation = (1, 1)  # was (4, 4)

    # model.backbone.layer4[2].conv2.padding = (1, 1)  # was (4, 4)
    # model.backbone.layer4[2].conv2.dilation = (1, 1)  # was (4, 4)

    return DeeplabV3_features(model, [3, 4, 6, 3], **kwargs)


def torchvision_resnet_weight_key_to_deeplab2(key: str) -> Optional[str]:
    segments = key.split('.')

    if segments[0].startswith('layer'):
        layer_num = int(segments[0].split('layer')[-1])
        dl_layer_num = layer_num + 1

        block_num = int(segments[1])
        dl_block_str = f'block{block_num + 1}'

        layer_type = segments[2]
        if layer_type == 'downsample':
            shortcut_module_num = int(segments[3])
            if shortcut_module_num == 0:
                module_name = 'conv'
            elif shortcut_module_num == 1:
                module_name = 'bn'
            else:
                raise ValueError(shortcut_module_num)

            return f'layer{dl_layer_num}.{dl_block_str}.shortcut.{module_name}.{segments[-1]}'

        else:
            layer_type, conv_num = segments[2][:-1], segments[2][-1]
            conv_num = int(conv_num)

            if conv_num == 1:
                dl_conv_name = 'reduce'
            elif conv_num == 2:
                dl_conv_name = 'conv3x3'
            elif conv_num == 3:
                dl_conv_name = 'increase'
            else:
                raise ValueError(conv_num)

            return f'layer{dl_layer_num}.{dl_block_str}.{dl_conv_name}.{layer_type}.{segments[-1]}'

    elif segments[0] in {'conv1', 'bn1'}:
        layer_type = segments[0][:-1]
        return f'layer1.conv1.{layer_type}.{segments[-1]}'

    return None


@gin.configurable(allowlist=['deeplab_n_features'])
def deeplabv2_resnet101_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED, **kwargs):
    model = DeepLabV2(
        n_classes=deeplab_n_features, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    return model
