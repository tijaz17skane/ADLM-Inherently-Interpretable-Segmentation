"""
Some constants related to Cityscapes and SUN datasets
"""

# Taken from deeplabv3 trained on COCO
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD = [0.229, 0.224, 0.225]

# Mapping of IDs to labels
# We follow https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# We merge 'void' classes into one but other classes stay the same

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


# Taken from deeplabv3 trained on COCO
SUN_MEAN = [0.485, 0.456, 0.406]
SUN_STD = [0.229, 0.224, 0.225]

# we follow https://rgbd.cs.princeton.edu/supp.pdf and use only 37 selected categories, all others are 'void'
SUN_CATEGORIES = [
    'void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
    'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
    'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
    'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
]

SUN_LABEL_2_ID = {k: i for i, k in enumerate(SUN_CATEGORIES)}

# some categories in the dataset have typos/different spelling
# this is an attempt to unify them
sun_convert_categories = {
    'night_stand': 'nightstand',
    'night stand': 'nightstand',
    'blind': 'blinds',
    'shower curtain': 'shower_curtain',
    'showercurtain': 'shower_curtain',
    'floormat': 'floor_mat',
    'floor mat': 'floor_mat',
    'floormats': 'floor_mat',
    'floor_mats': 'floor_mat',
    'floor mats': 'floor_mat'
}
