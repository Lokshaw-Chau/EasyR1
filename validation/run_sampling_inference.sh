#!/bin/bash

export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration
MODEL_PATH="/data2/share/Qwen3-VL-4B-Instruct"
DATA_PATH="/data1/zlx/workspace/EasyR1/data/data/AgentNetBench/agentnetbench_test_qwen3vl.parquet"
OUTPUT_PATH="./outputs"
NUM_ACTOR=4
NUM_SAMPLES=16  # Number of samples per query
TEMPERATURE=1.0
TOP_P=1.0
PREFIX="Action:"

echo "================================================"
echo "Multi-Sampling Inference Configuration"
echo "================================================"
echo "Model: $(basename $MODEL_PATH)"
echo "Data: $(basename $DATA_PATH)"
echo "Number of workers: $NUM_ACTOR"
echo "Samples per query: $NUM_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Prefix: $PREFIX"
echo "================================================"
echo ""

cd /data1/zlx/workspace/EasyR1/validation/guir1

python inference/inference_vllm_agentnetbench_sampling.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --num_actor ${NUM_ACTOR} \
    --num_samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --prefix ${PREFIX}

echo ""
echo "================================================"
echo "Sampling inference completed!"
echo "================================================"
