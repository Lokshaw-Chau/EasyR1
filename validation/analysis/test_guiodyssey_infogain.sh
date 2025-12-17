#!/bin/bash
#
# GUI-Odyssey InfoGain 计算 - 测试脚本
# 使用 10 个 episode 快速测试
#

# 设置环境变量 - 只使用 2 个 GPU 进行测试
export CUDA_VISIBLE_DEVICES=0,1

# 参数配置
DATA_DIR="/data1/zlx/workspace/EasyR1/data/data/GUIOdyssey"
MODEL_PATH="/data2/share/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="./guiodyssey_infogain_test"
NUM_EPISODES=10  # 测试只用 10 个 episode
NUM_GPUS=2       # 测试只用 2 个 GPU
MINI_BATCH_SIZE=4  # 减小 batch size
NUM_WORKERS=2      # 减小 worker 数
RANDOM_SEED=42

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行 InfoGain 计算
echo "=========================================="
echo "GUI-Odyssey InfoGain 计算 (测试模式)"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "Episode 数量: $NUM_EPISODES (测试)"
echo "GPU 数量: $NUM_GPUS"
echo "Batch Size: $MINI_BATCH_SIZE"
echo "=========================================="
echo ""

python calculate_infogain_guiodyssey.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$NUM_EPISODES" \
    --num_gpus "$NUM_GPUS" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --random_seed "$RANDOM_SEED"

echo ""
echo "=========================================="
echo "测试完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="
