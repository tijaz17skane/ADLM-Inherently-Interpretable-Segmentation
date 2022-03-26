# this scripts sets environment variables and activates conda environment
# run this script before any python script in this repository

# edit these variables if needed
export SOURCE_DATA_PATH='/shared/sets/datasets/sun_rgbd'
export DATA_PATH='/shared/sets/datasets/sun_rgbd/SUN_37_margin'
export LOG_DIR='./logs'

export RESULTS_DIR='/shared/results/sacha'
export USE_NEPTUNE=1


# these variables are unused in cityscapes segmentation
export TRAIN_DIR='/shared/sets/datasets/sun_rgbd'
export TRAIN_PUSH_DIR='/shared/sets/datasets/sun_rgbd'
export TEST_DIR='/shared/sets/datasets/sun_rgbd'

# set it if using neptune
# export NEPTUNE_API_TOKEN=''

# load all images to memory at once - makes training faster but uses lots of RAM
export LOAD_IMAGES_RAM=1


conda activate proto-seg
