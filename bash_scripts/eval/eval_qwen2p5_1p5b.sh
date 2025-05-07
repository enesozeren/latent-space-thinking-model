#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:2
#SBATCH --time=0-01:00:00
#SBATCH -o bash_outputs/output_qwen_chekpoint600_math500_eval.log
#SBATCH -e bash_outputs/error_qwen_chekpoint600_math500_eval.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Model details
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B"
MODEL_NAME_OR_PATH="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra32qov2/latent_reasoner_storage/outputs/qwen2p5_1p5b_rl/20250502_000432/checkpoint-600/"

# Output directory for results
OUTPUT_DIR="outputs"

# Evaluation parameters
DATASET="HuggingFaceH4/MATH-500"
BATCH_SIZE=16
MAX_LENGTH=1024
TEMPERATURE=0.7
TOP_P=0.95
SPLIT="test"
# NUM_EXAMPLES=64  # Set to specific number or remove this parameter to evaluate on all examples

echo "Starting evaluation of $MODEL_NAME_OR_PATH on $DATASET"

# Run evaluation script
CUDA_VISIBLE_DEVICES=0,1 python src/eval/eval.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR
    # --num_examples $NUM_EXAMPLES 

echo "Evaluation complete! Results saved to $OUTPUT_DIR"