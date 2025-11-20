#!/bin/bash

# 综合分析脚本：分析多次采样结果

set -e

# Configuration
SAMPLING_FILE="$1"
DATA_DIR="/data1/zlx/workspace/EasyR1/data/data/AgentNetBench/test_data"
AGENT_TYPE="qwen3vl"
MODEL_PATH="$2"

if [ -z "$SAMPLING_FILE" ]; then
    echo "Usage: $0 <sampling_file> [model_path]"
    echo ""
    echo "Example:"
    echo "  $0 outputs/Qwen3-VL-4B-Instruct/agentnetbench_test_qwen3vl_sampling16_Thought:.jsonl /data2/share/Qwen3-VL-4B-Instruct"
    exit 1
fi

if [ ! -f "$SAMPLING_FILE" ]; then
    echo "Error: Sampling file not found: $SAMPLING_FILE"
    exit 1
fi

echo "================================================"
echo "Multi-Sampling Result Analysis"
echo "================================================"
echo "Sampling file: $SAMPLING_FILE"
echo "Data directory: $DATA_DIR"
echo "Agent type: $AGENT_TYPE"
echo "================================================"
echo ""

# Step 1: Evaluate accuracy (pass@1, pass@k, avg acc)
echo "Step 1: Evaluating accuracy metrics..."
echo "--------------------------------------------"
python analyze_sampling_results.py \
    --sampling_file "$SAMPLING_FILE" \
    --data_dir "$DATA_DIR" \
    --agent_type "$AGENT_TYPE"

echo ""
echo "✓ Accuracy evaluation completed!"
echo ""

# Step 2: Calculate InfoGain (only if model path is provided)
if [ -n "$MODEL_PATH" ]; then
    echo "Step 2: Calculating InfoGain..."
    echo "--------------------------------------------"
    echo "Model: $MODEL_PATH"
    echo ""

    python calculate_infogain.py \
        --sampling_file "$SAMPLING_FILE" \
        --data_dir "$DATA_DIR" \
        --model_path "$MODEL_PATH"

    echo ""
    echo "✓ InfoGain calculation completed!"
    echo ""
else
    echo "Step 2: Skipped (no model path provided)"
    echo "To calculate InfoGain, provide model path as second argument"
    echo ""
fi

echo "================================================"
echo "Analysis completed!"
echo "================================================"
echo ""
echo "Results saved in:"
BASENAME=$(basename "$SAMPLING_FILE" .jsonl)
DIRNAME=$(dirname "$SAMPLING_FILE")
echo "  - $DIRNAME/analysis/${BASENAME}_analysis.json"
echo "  - $DIRNAME/analysis/${BASENAME}_metrics.json"
if [ -n "$MODEL_PATH" ]; then
    echo "  - $DIRNAME/analysis/${BASENAME}_infogain.json"
fi
echo ""
