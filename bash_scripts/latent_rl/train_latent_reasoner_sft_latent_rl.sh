#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:30:00
#SBATCH -o bash_outputs/output_latent_reasoner_1p5b_sft_latent_rl.log
#SBATCH -e bash_outputs/error_latent_reasoner_1p5b_sft_latent_rl.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/latent_rl/latent_reasoner_sft_latent_rl.yaml"
NUM_PROCESSES=1

# Launch training
echo "Starting Latent RL training"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/latent_rl/train.py --config $CONFIG_PATH