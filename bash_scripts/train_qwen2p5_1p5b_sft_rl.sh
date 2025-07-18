#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --time=0-10:00:00
#SBATCH -o bash_outputs/output_qwen2p5_1p5b_sft_rl.log
#SBATCH -e bash_outputs/error_qwen2p5_1p5b_sft_rl.log

# Activate environment & set PYTHONPATH (use official trl package instead of our modified one)
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/qwen2p5_1p5b_sft_rl.yaml"
NUM_PROCESSES=4

# Launch GRPO training
echo "Starting GRPO training"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/train/train_rl.py --config $CONFIG_PATH