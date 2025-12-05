#!/bin/bash
#
# 运行 GUI-Odyssey InfoGain 分析 notebook
#

echo "=========================================="
echo "GUI-Odyssey InfoGain Analysis"
echo "=========================================="
echo ""

# 激活虚拟环境
source /data1/zlx/workspace/EasyR1/.venv/bin/activate

# 切换到分析目录
cd /data1/zlx/workspace/EasyR1/validation/analysis

# 检查 InfoGain 结果文件是否存在
INFOGAIN_FILE="./guiodyssey_infogain_results/guiodyssey_infogain_results.json"
if [ ! -f "$INFOGAIN_FILE" ]; then
    echo "Error: InfoGain results file not found!"
    echo "Expected: $INFOGAIN_FILE"
    echo ""
    echo "Please run InfoGain calculation first:"
    echo "  bash run_guiodyssey_infogain.sh"
    exit 1
fi

echo "Found InfoGain results file: $INFOGAIN_FILE"
echo ""

# 检查必要的 Python 包
echo "Checking required packages..."
python -c "import jupyter, matplotlib, seaborn, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install jupyter matplotlib seaborn pandas numpy
fi

# 启动 Jupyter notebook
echo ""
echo "Starting Jupyter Notebook..."
echo "=========================================="
echo ""
echo "The notebook will:"
echo "  1. Visualize InfoGain distribution"
echo "  2. Analyze InfoGain by action types"
echo "  3. Select high-InfoGain balanced training set (1024 samples)"
echo "  4. Save results to: data/data/GUIOdyssey/train_high_infogain_1024.parquet (PARQUET format)"
echo ""
echo "=========================================="
echo ""

jupyter notebook analyze_guiodyssey_infogain.ipynb
