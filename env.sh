# this scripts sets environment variables and activates conda environment
# run this script before any python script in this repository

# edit these variables if needed
# SOURCE_DATA_PATH is the datapath to the CityScapes or Pancreas Dataset
export SOURCE_DATA_PATH='./data'
# DATA_PATH is the datapath to the CityScapes or Pancreas Dataset, this is more of a repetition. SOURCE_DATA_PATH barely gets used anywhere
export DATA_PATH='./data'
# LOG_DIR is for training logs
export LOG_DIR='./logs'
# DATA_PATH
export RESULTS_DIR='./results'


# set it if using neptune
# NEPTUNE for monitoring. This was a little tedious to get started with. beware of the latest neptune version. 
# Im using neptune-client           0.14.3

export USE_NEPTUNE=0  # set to 1 to use neptune
export NEPTUNE_API_TOKEN=""  # set your neptune api token
export NEPTUNE_PROJECT=""  # set your neptune project name

# please create this conda environment using 'env.yml'
conda activate proto-seg
