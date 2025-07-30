#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --time=0-07:00:00
#SBATCH -o bash_outputs/output_latent_reasoner_1p5b_sft_rl.log
#SBATCH -e bash_outputs/error_latent_reasoner_1p5b_sft_rl.log

# Activate environment & set PYTHONPATH
source activate latr_2
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/latent_reasoner_sft_rl.yaml"
NUM_PROCESSES=4

# Launch GRPO training
echo "Starting GRPO training"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/train/train_latent_reasoner_rl.py --config $CONFIG_PATH