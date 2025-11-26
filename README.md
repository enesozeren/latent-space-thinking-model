# Latent Space Thinking Model

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

Reinforcement Learning for Latent-Space Thinking in LLMs

Authors: Enes Özeren, Matthias Aßenmacher

This repository is built for implementation of latent-space thinking models, SFT + RL training methods, and evaluation.

## Repo Structure

```
latent-space-thinking-model/
├── bash_scripts/                    # Shell scripts for training and evaluation
│   ├── eval/                        # Model evaluation scripts
│   ├── latent_rl/                   # Latent RL training scripts
│   ├── train_grpo/                  # GRPO training scripts
│   ├── train_sft/                   # Supervised fine-tuning scripts
│   └── value_model/                 # Value model scripts
├── notebooks/                        # Jupyter notebooks for analysis
├── outputs/                          # Training outputs and checkpoints
├── prompts/                          # Prompt templates and utilities
├── src/                             # Source code
│   ├── configs/                     # Configuration files
│   │   ├── latent_rl/               # Latent RL configurations
│   │   ├── latr/                    # LATR configurations
│   │   ├── qwen/                    # Qwen model configurations
│   │   ├── qwen_base/               # Base Qwen configurations
│   │   └── value_model/             # Value model configurations
│   ├── data_process/                # Data processing utilities
│   │   ├── check_data_contemination.py  # Check for data contamination
│   │   ├── process_data_eval.py         # Process evaluation data
│   │   └── process_data.py             # General data processing
│   ├── eval/                        # Evaluation framework
│   │   ├── eval_utils.py
│   │   └── eval.py
│   ├── latent_reasoner/             # LatR implementation
│   │   ├── inference.py
│   │   └── model.py
│   ├── latent_rl/                   # Latent RL training
│   │   ├── latent_rl_trainer.py
│   │   └── train.py
│   ├── train_sft_grpo/              # SFT and GRPO training
│   │   ├── lightning_modules.py
│   │   ├── rewards.py
│   │   ├── train_grpo.py
│   │   ├── train_latent_reasoner_grpo.py
│   ├── train_sft.py                 # SFT training
│   │   └── utils.py
│   └── value_model/                 # Value model implementation
│       ├── create_value_model_train_data.py
│       ├── lightning_modules.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── requirements.txt
└── README.md
```

## How to Run the Code

### Environment Set Up
1. Download python version 3.12.9
2. Create a conda environment
3. Activate the conda environment and install the requirements: `pip install -r requirements.txt`

### Running Scripts
To run the scripts, check out the bash commands in the `bash_scripts` directory.

Note that the bash scripts are designed to run the python scripts in the LRZ and MCML GPU clusters using slurm.

Before runing the bash scripts, you can adjust the parameters in the config files that are pointed.

#### LatR Models

- To train the LatR 1.5B SFT model use:
    ```bash
    bash bash_scripts/train_sft/train_latent_reasoner_sft.sh
    ```

- To train the LatR 1.5B SFT & GRPO model:
    You need a modified version of trl package. You can install this with `pip install git+https://github.com/enesozeren/trl.git`
    ```bash
    bash bash_scripts/train_grpo/train_latent_reasoner_sft_grpo.sh
    ```

- To train the LatR 1.5B SFT & Latent RL model:
    1. First create the dataset for value model training. You need the LatR SFT model for this process.
    ```bash
    bash bash_scripts/value_model/create_value_model_training_data.sh
    ```
    2. Train the value model with:
    ```bash
    bash bash_scripts/value_model/train_value_model_head.sh
    ```
    3. Train the policy model with:
    ```bash
    bash bash_scripts/latent_rl/train_latent_reasoner_sft_latent_rl.sh
    ```

#### Qwen Models

- To train Qwen 1.5B SFT use:
    ```bash
    bash bash_scripts/train_sft/train_qwen2p5_1p5b_sft.sh
    ```

- To train the Qwen 1.5B SFT & GRPO model:
    First `pip install trl==018.0`
    Then run the following bash script.
    ```bash
    bash bash_scripts/train_grpo/train_qwen2p5_1p5b_sft_grpo.sh
    ```

#### Evaluation

- Use the corresponding bash script in the `bash_scripts/eval` directory.
    For example to evaluate the Qwen 1.5B base model on GSM8K benchmark use:
    ```bash
    bash bash_scripts/eval/eval_model_1.sh
    ```
