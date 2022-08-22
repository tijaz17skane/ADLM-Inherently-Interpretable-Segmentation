# this scripts sets environment variables and activates conda environment
# run this script before any python script in this repository

# edit these variables if needed
export SOURCE_DATA_PATH='/media/mikolaj/SSD/ml_data/isbi_2012'
export DATA_PATH='/media/mikolaj/SSD/ml_data/isbi_2012_trainval'
export LOG_DIR='./logs'

export RESULTS_DIR='/media/mikolaj/HDD/proto_segmentation'
export USE_NEPTUNE=1

export TRAIN_DIR='/media/mikolaj/SSD/ml_data/isbi_2012_trainval'
export TRAIN_PUSH_DIR='/media/mikolaj/SSD/ml_data/isbi_2012_trainval'
export TEST_DIR='/media/mikolaj/SSD/ml_data/isbi_2012_trainval'


# set it if using neptune
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MjRjODA1MS03MmIxLTQyZDAtYjgwYi0xOGM4ZWZjNjI4YzcifQ=="

conda activate proto-seg
