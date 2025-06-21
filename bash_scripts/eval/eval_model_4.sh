#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:45:00
#SBATCH -o bash_outputs/output_eval_4.log
#SBATCH -e bash_outputs/error_eval_4.log

# Activate environment & set PYTHONPATH
source /dss/dsshome1/0B/ra32qov2/anaconda3/etc/profile.d/conda.sh
conda activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

CONFIG_PATH="src/configs/base_qwen2p5_1p5b_eval_gsm.yaml"

echo "Starting evaluation"
CUDA_VISIBLE_DEVICES=0 python src/eval/eval.py \
    --config $CONFIG_PATH

echo "Evaluation complete!"