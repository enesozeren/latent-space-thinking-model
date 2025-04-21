#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4                # 1 for vLLM, 3 for training
#SBATCH --time=0-04:00:00
#SBATCH -o bash_outputs/output_gemma_3_1b_it_rl.log
#SBATCH -e bash_outputs/error_gemma_3_1b_it_rl.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/gemma_3_1b_it_rl.yaml"
NUM_PROCESSES=3

# 1) Launch vLLM server on GPU 0 and 1
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model google/gemma-3-1b-it &
VLLM_PID=$!

# Give the server some time to initialize (adjust if needed)
sleep 3m

# 2) Launch GRPO training on GPUs 1 & 2
echo "Starting GRPO training on GPUs 1,2,3..."
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/train/train_rl.py --config $CONFIG_PATH

# 3) Clean up vLLM server
echo "Training complete — shutting down vLLM server (PID $VLLM_PID)..."
kill $VLLM_PID