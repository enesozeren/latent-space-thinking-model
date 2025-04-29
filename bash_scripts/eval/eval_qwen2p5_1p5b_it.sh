#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:2
#SBATCH --time=0-01:30:00
#SBATCH -o bash_outputs/output_qwen_it_gsm8k_eval.log
#SBATCH -e bash_outputs/error_qwen_it_gsm8k_eval.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Model details
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TYPE="it"  # Instruction-tuned model

# Output directory for results
OUTPUT_DIR="outputs"

# Evaluation parameters
DATASET="gsm8k"
BATCH_SIZE=64
MAX_LENGTH=4096
TEMPERATURE=0.2
TOP_P=0.95
SPLIT="test"
# NUM_EXAMPLES=256  # Set to specific number or remove this parameter to evaluate on all examples

echo "Starting evaluation of $MODEL_NAME on $DATASET"

# Run evaluation script
CUDA_VISIBLE_DEVICES=0,1 python src/eval/eval_gsm8k.py \
    --model_name_or_path $MODEL_NAME \
    --dataset $DATASET \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR \
    --device_map "auto"
    # --num_examples $NUM_EXAMPLES 

echo "Evaluation complete! Results saved to $OUTPUT_DIR"