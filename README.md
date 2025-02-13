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
#### Medical Decathlon Segmentation Dataset
Modify paths according to your downloads. and If you're using a dataset other than Pancreas.
The following commands will convert the scans into numpy slices, resize and normalize them, convert images.npy to pngs and make train, test and validation splits.


You may edit the preprocessPancreasScans.py to skip the conversion to png

Don't forget to replace contents of segmentation/constants.py with segmentation/constants1.py when going with MDS

Then generate a list of the preprocessed images in the splits.
```
python -m segmentation.preprocessPancreasScans /path/to/datasets/task07_pancreas /path/to/data
python -m segmentation.generateImageList /path/to/img_with_margin_0 /path/to/data
```

## Training model


### warmup + joint training + prototype push
#### CityScapes
```
# train on Cityscapes, no similarity loss, ImageNet pretraining
python -m segmentation.train cityscapes_no_kld_imnet <your_training_run_name>

# train on Cityscapes, with similarity loss, COCO pretraining
python -m segmentation.train cityscapes_kld_coco <your_training_run_name>

# train on Cityscapes, with similarity loss, ImageNet pretraining
python -m segmentation.train cityscapes_kld_imnet <your_training_run_name>
```
#### Medical Decathlon Segmentation Dataset
```
# train on Medical Decathlon for Pancreas
python -m segmentation.train mds_new <your_training_run_name>
```

### Pruning and finetuning after pruning
#### CityScapes
```
python -m segmentation.run_pruning cityscapes_kld_imnet <your_training_run_name>
python -m segmentation.train cityscapes_kld_imnet <your_training_run_name> --pruned
```
#### Medical Decathlon Segmentation Dataset
```
python -m segmentation.run_pruning mds_new <your_training_run_name>
python -m segmentation.train mds_new <your_training_run_name> --pruned
```

## Evaluation
#### CityScapes
```
# The evaluation saves mIOU results in model directory. It also generates a few additional plots.
# <training_staged> should be one of: 'warmup', 'nopush', 'push', 'pruned'

python -m segmentation.eval_valid <your_cityscapes_training_run_name> <training_stage>
python -m segmentation.eval_test <your_cityscapes_training_run_name> <training_stage>
```
#### Medical Decathlon Segmentation Dataset
```
# The evaluation saves mIOU results in model directory. It also generates a few additional plots.
# <training_staged> should be one of: 'warmup', 'nopush', 'push', 'pruned'

python -m segmentation.eval_valid <your_mds_training_run_name> <training_stage>
python -m segmentation.eval_test <your_mds_training_run_name> <training_stage>
```


# U-Noise for Inherently Interpretable Segmentation for Medical Decathlon Dataset - Pancreas

[**U-Noise: Learnable Noise Masks for Interpretable Image
Segmentation**](https://arxiv.org/abs/2101.05791)<br>
Teddy Koker, Fatemehsadat Mireshghallah, Tom Titcombe, Georgios Kaissis

## Download Data/Pre-trained Models

The dataset can be created by downloading and un-taring
[Task07_Pancreas.tar](https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view?usp=sharing)
from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) into the
`data/` directory. Once there, run the `prepare_data.py` script within the
directory.

```bash
./download.sh
```

This will download the following files:
```
models
├── unoise_small.ckpt   # small U-Noise model
├── unoise_medium.ckpt  # medium U-Noise model
├── unoise_large.ckpt   # large U-Noise model TODO
├── unoise_small_pretrained.ckpt  # small U-Noise model pretrained
├── unoise_medium_pretrained.ckpt # medium U-Noise model pretrained
├── unoise_large_pretrained.ckpt  # large U-Noise model pretrained
└── utility.ckpt        # pretrained utility model
```

Alternatively, each model can be downloaded individually:

*Note: each U-Noise model contains a copy of the utility model*

Model          | # parameters | weights
---            | ---          | ---
Utility        | 34M          | [utility.ckpt](https://drive.google.com/file/d/1uXsgpJSOiKfIe1haqoRchx-AMRF9ormK/view?usp=sharing)
U-Noise Small  | 10K          | [unoise_small.ckpt](https://drive.google.com/file/d/1FEy61tSQzYF10e0N8xNENs0a0Rv0UMPv/view?usp=sharing)
U-Noise Medium | 130K         | [unoise_medium.ckpt](https://drive.google.com/file/d/11_rTHLkB56QIlPXTlRb7ln9WURbbAUDD/view?usp=sharing)
U-Noise Large  | 537K         | [unoise_large.ckpt](https://drive.google.com/file/d/1evV2daEgnfbyctwCkQ5PHXhZT8LNsidr/view?usp=sharing)
U-Noise Small  (pretrained)| 10K  | [unoise_small_pretrained.ckpt](https://drive.google.com/file/d/1kzR1I_lgynPtqEQqwaHiVdmQKghSi2bv/view?usp=sharing)
U-Noise Medium (pretrained)| 130K | [unoise_medium_pretrained.ckpt](https://drive.google.com/file/d/1xdJH9jcRZoVa6i_mdKCfbjyPQQrMbLac/view?usp=sharing)
U-Noise Large (pretrained) | 537K | [unoise_large_pretrained.ckpt](https://drive.google.com/file/d/1834JqlUcxeS3ifAnTjiCGHvP3GYST7Bl/view?usp=sharing)

U-Nets       | Params | Depth | Channels
---          | ---    | ---   | ---
Utility      | 34M    | 5     | 1, 64, 128, 256, 512, 1024, 512, ...
Small        | 28K    | 2     | 1, 16, 32, 16, 1
Medium       | 130K   | 3     | 1, 16, 32, 64, 32, 16, 1
Medium       | 537K   | 4     | 1, 16, 32, 64, 128, 64, 32, 16, 1

## Reproducing Results

Train **Utility** model (~3 hours on NVIDIA RTX 3090):
```bash
python src/train_util.py
```

Train **U-Noise Small**:
```bash
python src/train_noise.py --depth 2 --channel_factor 4 --batch_size 8
```           

Train **U-Noise Medium**:
```bash
python src/train_noise.py --depth 3 --channel_factor 4 --batch_size 8
```            

Train **U-Noise Large**:
```bash
python src/train_noise.py --depth 4 --channel_factor 4 --batch_size 8
```            

Train **U-Noise Small (Pretrained)**:
```bash
python src/train_util.py --depth 2 --channel_factor 4 --batch_size 8
python src/train_noise.py --depth 2 --channel_factor 4 --batch_size 8 \
 --pretrained /path/to/pretrained --learning_rate 1e-3
```            

Train **U-Noise Medium (Pretrained)**:
```bash
python src/train_util.py --depth 3 --channel_factor 4 --batch_size 8
python src/train_noise.py --depth 3 --channel_factor 4 --batch_size 8 \
 --pretrained /path/to/pretrained --learning_rate 1e-3
```            

Train **U-Noise Large (Pretrained)**:
```bash
python src/train_util.py --depth 4 --channel_factor 4 --batch_size 8
python src/train_noise.py --depth 4 --channel_factor 4 --batch_size 8 \
 --pretrained /path/to/pretrained --learning_rate 1e-3
```            

## Citation

```bibtex
@misc{koker2021unoise,
      title={U-Noise: Learnable Noise Masks for Interpretable Image Segmentation}, 
      author={Teddy Koker and Fatemehsadat Mireshghallah and Tom Titcombe and Georgios Kaissis},
      year={2021},
      eprint={2101.05791},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```           
