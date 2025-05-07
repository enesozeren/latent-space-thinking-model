#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4                # 1 for vLLM, 3 for training
#SBATCH --time=0-08:00:00
#SBATCH -o bash_outputs/output_qwen2p5_3b_rl_openr1data.log
#SBATCH -e bash_outputs/error_qwen2p5_3b_rl_openr1data.log

# Activate environment & set PYTHONPATH
source activate latr
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0B/ra32qov2/latent-reasoner

# Config and number of processes for training
CONFIG_PATH="src/configs/qwen2p5_3b_rl.yaml"
NUM_PROCESSES=3

# 1) Launch vLLM server
echo "Starting vLLM server"
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model Qwen/Qwen2.5-3B \
    --tensor-parallel-size 1 \
    --gpu_memory_utilization 0.9 \
    --max-model-len 3840 & 
    VLLM_PID=$!

# Give the server some time to initialize (adjust if needed)
sleep 3m

# 2) Launch GRPO training
echo "Starting GRPO training"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num_processes $NUM_PROCESSES \
    src/train/train_rl.py \
    --config $CONFIG_PATH
    
# 3) Clean up vLLM server
echo "Training complete — shutting down vLLM server (PID $VLLM_PID)..."
kill $VLLM_PID