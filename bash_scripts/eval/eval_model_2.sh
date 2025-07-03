#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --time=0-00:30:00
#SBATCH -o bash_outputs/output_eval_2.log
#SBATCH -e bash_outputs/error_eval_2.log

# Activate environment & set PYTHONPATH
source /dss/dsshome1/0B/ra32qov2/anaconda3/etc/profile.d/conda.sh
conda activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

CONFIG_PATH="src/configs/latent_reasoner_sft_eval_gsm.yaml"

echo "Starting evaluation"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    src/eval/eval.py --config_path $CONFIG_PATH

echo "Evaluation complete!"