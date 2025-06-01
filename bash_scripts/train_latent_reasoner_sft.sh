#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --time=0-01:00:00
#SBATCH -o bash_outputs/output_latent_reasoner_1p5b_sft.log
#SBATCH -e bash_outputs/error_latent_reasoner_1p5b_sft.log

# Activate environment & set PYTHONPATH
source activate latr_2
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/other-repos/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/latent_reasoner_sft.yaml"
NUM_PROCESSES=4

# Launch GRPO training
echo "Starting SFT for Latent Reasoner"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/train/train_latent_reasoner_sft.py --config $CONFIG_PATH