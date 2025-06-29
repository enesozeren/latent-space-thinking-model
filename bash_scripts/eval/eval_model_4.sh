#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --time=0-02:00:00
#SBATCH -o bash_outputs/output_eval_4.log
#SBATCH -e bash_outputs/error_eval_4.log

# Activate environment & set PYTHONPATH
source /dss/dsshome1/0B/ra32qov2/anaconda3/etc/profile.d/conda.sh
conda activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

CONFIG_PATH="src/configs/qwen2p5_1p5b_sft_eval_math.yaml"

echo "Starting evaluation"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    src/eval/eval.py --config_path $CONFIG_PATH

echo "Evaluation complete!"