#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
MODEL_PATH=/your/path/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=2,3,4,7

python3 -m verl.trainer.main \
    config=examples/configs/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.reward_type=sequential \
    trainer.n_gpus_per_node=4 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=128 \
    data.format_prompt=adaptive \
    worker.rollout.rollout_intervention=true \
    algorithm.think_alpha=0.001 \
    algorithm.use_kl_loss=false \
    algorithm.disable_kl=true \
    worker.reward.reward_function=./examples/reward_function/gui_adaptive.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_baseline_adaptive_no_kl_thinkless \
    trainer.save_checkpoint_path=/your/path/EasyR1/ckpts/qwen2_5_vl_3b_baseline_adaptive_no_kl_thinkless
