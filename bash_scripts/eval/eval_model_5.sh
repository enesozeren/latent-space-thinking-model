#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:45:00
#SBATCH -o bash_outputs/output_eval_5.log
#SBATCH -e bash_outputs/error_eval_5.log

# Activate environment & set PYTHONPATH
source /dss/dsshome1/0B/ra32qov2/anaconda3/etc/profile.d/conda.sh
conda activate latr_2
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

CONFIG_PATH="src/configs/latent_reasoner_sft_eval_gsm.yaml"

echo "Starting evaluation"

CUDA_VISIBLE_DEVICES=0 python src/eval/eval.py --config $CONFIG_PATH

echo "Evaluation complete!"