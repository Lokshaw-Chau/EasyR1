#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
SYSTEM_PROMPT=""""""
MODEL_PATH=/mnt/data/home/zoulexiao/workspace/LLaMA-Factory/saves/ada_think/sft  # replace it with your local file path

export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=2,3,4,5

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/data/share/data1/gui-r1/train.parquet \
    data.val_files=/mnt/data/share/data1/gui-r1/test.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/gui.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_guir1_ton \
    trainer.n_gpus_per_node=4 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=128