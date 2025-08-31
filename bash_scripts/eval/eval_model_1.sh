#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH -o bash_outputs/output_eval_1.log
#SBATCH -e bash_outputs/error_eval_1.log

# Activate environment & set PYTHONPATH
source /dss/dsshome1/0B/ra32qov2/anaconda3/etc/profile.d/conda.sh
conda activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

CONFIG_PATH="src/configs/qwen_base/base_qwen2p5_1p5b_eval_gsm.yaml"

echo "Starting evaluation"

CUDA_VISIBLE_DEVICES=0 python src/eval/eval.py --config $CONFIG_PATH

echo "Evaluation complete!"