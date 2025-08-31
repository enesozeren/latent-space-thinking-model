#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH -o bash_outputs/output_value_mode.log
#SBATCH -e bash_outputs/error_value_model.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/value_model/value_model_training.yaml"
NUM_PROCESSES=1

# Launch training
echo "Starting value model training"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/value_model/train.py --config $CONFIG_PATH