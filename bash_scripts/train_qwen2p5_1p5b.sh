#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:8                # 2 for vLLM, 6 for training
#SBATCH --time=0-03:00:00
#SBATCH -o bash_outputs/output_qwen2p5_1p5b_rl_8gpu.log
#SBATCH -e bash_outputs/error_qwen2p5_1p5b_rl_8gpu.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/qwen2p5_1p5b_rl.yaml"
NUM_PROCESSES=6

# 1) Launch vLLM server
echo "Starting vLLM server"
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model Qwen/Qwen2.5-1.5B --tensor-parallel-size 2 &
VLLM_PID=$!

# Give the server some time to initialize (adjust if needed)
sleep 3m

# 2) Launch GRPO training
echo "Starting GRPO training"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/train/train_rl.py --config $CONFIG_PATH

# 3) Clean up vLLM server
echo "Training complete — shutting down vLLM server (PID $VLLM_PID)..."
kill $VLLM_PID