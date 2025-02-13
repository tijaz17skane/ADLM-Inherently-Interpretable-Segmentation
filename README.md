# Semantic Segmentation with Prototypes
This repository is a fork of https://github.com/cfchen-duke/ProtoPNet, adapted for ProtoSeg.

We also use pytorch implemention of DeepLab from: https://github.com/kazuto1011/deeplab-pytorch

Since I implemented this using CityScapes Dataset only, I did not have to follow the guide for PASCAL VOC. Download the Coco checkpoints form the link above and you're good to go. 


## Setting environment

Install conda environment:
`conda install env -f env.yml`

Before any script run:
`source env.sh`


## Setting up data

### Downloading data

Please use the following links to download Cityscapes:
- Cityscapes: https://www.cityscapes-dataset.com/downloads/ (gtFine_trainvaltest.zip and leftImgbit_trainvaltest.zip are needed)
- Medical Decathlon (Task 07- Pancreas): http://medicaldecathlon.com/dataaws/

#### I Used the following directory structure:

# ProtoSegFolder
```
.
ProtoSegFolder
├── proto-segmentation
│   ├── deeplab_pytorch
│   └── all other elements of the cloned repository
└── datasets
    ├── cityscapes
    │   ├── gtfine
    │   └── leftImg8Bit
    └── task07_pancreas
        ├── ImagesTs
        ├── ImagesTr
        ├── labelsTr
        └── etc
```
### Preprocessing data
We convert images to .npy files for faster loading during training. To do this, run the following commands:
#### CityScapes
```
# replace <N> with the number of parallel jobs
python -m segmentation.preprocess_cityscapes <N> # for cityscapes
python -m segmentation.preprocess_cityscapes preprocess-cityscapes-obj-masks <N>
```
like this:
```
python -m segmentation.preprocess_cityscapes preprocess-cityscapes 4
python -m segmentation.preprocess_cityscapes preprocess-cityscapes-obj-masks 4
python -m segmentation.img_to_numpy
```
#### Medical Decathlon Dataset
Modify paths according to your downloads. and If you're using a dataset other than Pancreas.
The following commands will convert the scans into numpy slices, resize and normalize them, convert images.npy to pngs and make train, test and validation splits.
Then generate a list of the preprocessed images in the splits.
```
python -m segmentation.ProcessPancreasScans /path/to/datasets/task07_pancreas /path/to/data
python-m segmentation.GenerateImageList.py /path/to/img_with_margin_0 /path/to/data

```

## Training model


### warmup + joint training + prototype push
```
# train on Cityscapes, no similarity loss, ImageNet pretraining
python -m segmentation.train cityscapes_no_kld_imnet <your_training_run_name>

# train on Cityscapes, with similarity loss, COCO pretraining
python -m segmentation.train cityscapes_kld_coco <your_training_run_name>

# train on Cityscapes, with similarity loss, ImageNet pretraining
python -m segmentation.train cityscapes_kld_imnet <your_training_run_name>

# train on Medical Decathlon for Pancreas
python -m segmentation.train mds_new <your_training_run_name>
```

### pruning and finetuning after pruning
```
python -m segmentation.run_pruning pascal_kld_imnet <your_training_run_name>
python -m segmentation.train pascal_kld_imnet <your_training_run_name> --pruned
```

## Evaluation
```
# Evaluation on valid set for Cityscapes ('cityscapes_kld_imnet' should be replaced with your training config)
# The evaluation saves mIOU results in model directory. It also generates a few additional plots.
# <training_staged> should be one of: 'warmup', 'nopush', 'push', 'pruned'
python -m segmentation.eval_valid <your_training_run_name> <training_stage>

# For evaluating on Pascal dataset, add '-p' flag
python -m segmentation.eval_valid <your_training_run_name> <training_stage> -p

# Generating predictions on cityscapes or pascal test set:
python -m segmentation.eval_test <your_cityscapes_training_run_name> <training_stage>
python -m segmentation.eval_test <your_pascal_training_run_name> <training_stage> -p
```
