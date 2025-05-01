#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:2
#SBATCH --time=0-01:15:00
#SBATCH -o bash_outputs/output_qwen_rl_gsm8k_eval_checkpoint200.log
#SBATCH -e bash_outputs/error_qwen_rl_gsm8k_eval_checkpoint200.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Model details
MODEL_PATH="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra32qov2/latent_reasoner_storage/outputs/qwen2p5_1p5b_rl/20250427_232106/checkpoint-200/"

# Output directory for results
OUTPUT_DIR="outputs"

# Evaluation parameters
DATASET="gsm8k"
BATCH_SIZE=16
MAX_LENGTH=1024
TEMPERATURE=0.2
TOP_P=0.95
SPLIT="test"
# NUM_EXAMPLES=64  # Set to specific number or remove this parameter to evaluate on all examples

echo "Starting evaluation of $MODEL_PATH on $DATASET"

# Run evaluation script
CUDA_VISIBLE_DEVICES=0,1 python src/eval/eval_gsm8k.py \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR
    # --num_examples $NUM_EXAMPLES 

echo "Evaluation complete! Results saved to $OUTPUT_DIR"