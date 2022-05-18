import torchvision
from torch import nn

from deeplab_pytorch.libs.models import DeepLabV2_ResNet101_MSC
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


def deeplabv2_resnet101_features(pretrained=False, **kwargs):
    model = DeepLabV2(
        n_classes=64, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )

    model.conv_info = lambda: ([], [], [])

    return model
