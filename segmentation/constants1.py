"""
Some constants related to Cityscapes and PASCAL datasets
"""

# Taken from deeplabv2 trained on ImageNet
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD = [0.229, 0.224, 0.225]

# Mapping of IDs to labels
# We follow https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# We merge 'void' classes into one but other classes stay the same

'''
CITYSCAPES_ID_2_LABEL = {
    0: 'void',
    1: 'void',
    2: 'void',
    3: 'void',
    4: 'void',
    5: 'void',
    6: 'void',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    -1: 'license plate'
}

CITYSCAPES_CATEGORIES = [
    'void',
    'road',
    'sidewalk',
    'parking',
    'rail track',
    'building',
    'wall',
    'fence',
    'guard rail',
    'bridge',
    'tunnel',
    'pole',
    'polegroup',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'caravan',
    'trailer',
    'train',
    'motorcycle',
    'bicycle',
    'license plate'
]

CITYSCAPES_19_EVAL_CATEGORIES = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 0,
    5: 3,
    6: 4,
    7: 5,
    8: 0,
    9: 0,
    10: 0,
    11: 6,
    12: 0,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    17: 11,
    18: 12,
    19: 13,
    20: 14,
    21: 15,
    22: 16,
    23: 0,
    24: 0,
    25: 17,
    26: 18,
    27: 19,
    28: 0,
}
'''

CITYSCAPES_ID_2_LABEL = {
    0: 'void',
    1: 'pancreas',
    2: 'tumour'
}

CITYSCAPES_CATEGORIES = [
    'void',
    'pancreas',
    'tumour'
]

CITYSCAPES_19_EVAL_CATEGORIES = {
    0: 0,
    1: 1,
    2: 2
}

PASCAL_ID_MAPPING = {
    255: 0,
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21
}

PASCAL_CATEGORIES = [
    'void',
    '__background__',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
