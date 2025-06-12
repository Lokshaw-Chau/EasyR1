
export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=2,3,4,5,7

# for ckpt_num in sft sft/checkpoint-85 sft/checkpoint-170 sft/checkpoint-255 sft/checkpoint-340
# do 

MODEL_PATH=/mnt/data/home/zoulexiao/workspace/LLaMA-Factory/saves/ada_think/official_prompt

DATA_DIR=/mnt/data/share/data1/gui-r1


# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_high_test.parquet
# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_low_test.parquet
# python inference/inference_vllm_guiact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiact_web_test.parquet
# python inference/inference_vllm_guiodyssey.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiodyssey_test.parquet
# python inference/inference_vllm_omniact_desktop.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_desktop_test.parquet
# python inference/inference_vllm_omniact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_web_test.parquet
python inference/inference_vllm_uground_official.py --model_path ${MODEL_PATH} --data_path /mnt/data/share/data1/Uground-V1-Data-Box/ATtrain/uground_sampled.parquet --num_actor 5
# python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor 4

# done