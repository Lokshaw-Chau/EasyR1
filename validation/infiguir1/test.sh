export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=4,5,6,7


python android_control.py \
    --model_path /mnt/data/share/Qwen2.5-VL-3B-Instruct \
    --eval_type high \
    --eval_file /mnt/data/share/data1/android_control_test_infigui-r1/android_control_test.json \
    --image_root /mnt/data/share/data1/android_control_test_infigui-r1 \
    --output_dir ./outputs \
    --thinking \
    --debug