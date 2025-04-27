#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH --gres=gpu:4                # 1 for vLLM, 3 for training
#SBATCH --time=0-02:00:00
#SBATCH -o bash_outputs/output_qwen2p5_1p5b_rl_chat.log
#SBATCH -e bash_outputs/error_qwen2p5_1p5b_rl_chat.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/qwen2p5_1p5b_rl.yaml"
NUM_PROCESSES=3

# 1) Launch vLLM server on GPU 0 with chat template support
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B &
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