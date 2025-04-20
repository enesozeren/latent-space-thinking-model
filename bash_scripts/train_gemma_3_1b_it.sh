#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:30:00
#SBATCH -o bash_outputs/output_gemma_3_1b_it_rl.log
#SBATCH -e bash_outputs/error_gemma_3_1b_it_rl.log

# Set environment variables
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Default configuration
CONFIG_PATH="src/configs/gemma_3_1b_it_rl.yaml"
NUM_PROCESSES=2

# Launch training with Hugging Face Accelerate
accelerate launch --num_processes $NUM_PROCESSES src/train/train_rl.py --config $CONFIG_PATH

echo "Training complete!"