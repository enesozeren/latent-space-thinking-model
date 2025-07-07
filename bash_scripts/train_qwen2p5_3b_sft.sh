#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --time=0-02:00:00
#SBATCH -o bash_outputs/output_qwen_3b_sft.log
#SBATCH -e bash_outputs/error_qwen_3b_sft.log

# Activate environment & set PYTHONPATH
source activate latr_2
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config path
CONFIG_PATH="src/configs/qwen2p5_3b_sft.yaml"

# Launch Lightning training
echo "Starting SFT with PyTorch Lightning"
python src/train/train_sft.py --config $CONFIG_PATH
