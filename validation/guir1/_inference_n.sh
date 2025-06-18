
export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=2,6

# for ckpt_num in sft sft/checkpoint-85 sft/checkpoint-170 sft/checkpoint-255 sft/checkpoint-340
# do 

MODEL_PATH=/root/workspace/datasets/share/share1/Qwen2.5-VL-3B-Instruct

DATA_DIR=/root/workspace/datasets/gui-r1


python inference/inference_vllm_pass_at_n_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor 2

# done