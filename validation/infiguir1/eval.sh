export NCCL_P2P_LEVEL=NVL

MODEL_PATH=your_path
DATA_DIR=your_path

CUDA_VISIBLE_DEVICES=4 python android_control_eval.py \
    --model_path $MODEL_PATH \
    --eval_type high \
    --eval_file $DATA_DIR/android_control_test.json \
    --image_root $DATA_DIR \
    --output_dir ./outputs \
    --thinking \
    --split 1 \
    --total_split 60