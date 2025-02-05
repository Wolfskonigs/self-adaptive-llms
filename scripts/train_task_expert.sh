# !/bin/bash

# Task Selection
TASK="chatbot" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=5

# This script needs 2 gpus
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=llama3i8b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS