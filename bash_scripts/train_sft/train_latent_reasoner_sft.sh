#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:8
#SBATCH --time=0-03:30:00
#SBATCH -o bash_outputs/output_latent_reasoner_1p5b_sft_2.log
#SBATCH -e bash_outputs/error_latent_reasoner_1p5b_sft_2.log

# Activate environment & set PYTHONPATH
source activate latr_2
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config path
CONFIG_PATH="src/configs/latr/latent_reasoner_sft.yaml"

# Launch Lightning training
echo "Starting SFT for Latent Reasoner with PyTorch Lightning"
python src/train_sft_grpo/train_sft.py --config $CONFIG_PATH
