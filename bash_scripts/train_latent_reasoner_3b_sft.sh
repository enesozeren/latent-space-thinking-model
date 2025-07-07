#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --time=0-10:00:00
#SBATCH -o bash_outputs/output_latent_reasoner_3b_sft.log
#SBATCH -e bash_outputs/error_latent_reasoner_3b_sft.log

# Activate environment & set PYTHONPATH
source activate latr_2
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config path
CONFIG_PATH="src/configs/latent_reasoner_3b_sft.yaml"

# Launch Lightning training
echo "Starting SFT for Latent Reasoner with PyTorch Lightning"
python src/train/train_sft.py --config $CONFIG_PATH
