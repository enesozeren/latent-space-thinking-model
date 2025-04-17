#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --time=0-00:30:00
#SBATCH -o bash_outputs/output_gemma_3_1b_it_rl_2.log
#SBATCH -e bash_outputs/error_gemma_3_1b_it_rl_2.log

# Set environment variables
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Default configuration
CONFIG_PATH="src/configs/gemma_3_1b_it_rl.yaml"
NUM_PROCESSES=2

# Launch training with Hugging Face Accelerate
accelerate launch --num_processes $NUM_PROCESSES src/train/train_rl.py --config $CONFIG_PATH

echo "Training complete!"