#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH -o bash_outputs/value_model_training_data_creation.log
#SBATCH -e bash_outputs/value_model_training_data_creation.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes
CONFIG_PATH="src/configs/value_model/value_model_training_data_creation.yaml"
NUM_PROCESSES=1

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/value_model/create_value_model_train_data.py --config $CONFIG_PATH