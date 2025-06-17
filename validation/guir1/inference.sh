
export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=0,1

# for ckpt_num in sft sft/checkpoint-85 sft/checkpoint-170 sft/checkpoint-255 sft/checkpoint-340
# do 

MODEL_PATH=/root/workspace/EasyR1/ckpts/qwen2_5_vl_3b_guir1_exp2_0615/global_step_120/actor/huggingface

DATA_DIR=/root/workspace/datasets/gui-r1


# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_high_test.parquet
# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_low_test.parquet
# python inference/inference_vllm_guiact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiact_web_test.parquet
# python inference/inference_vllm_guiodyssey.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiodyssey_test.parquet
# python inference/inference_vllm_omniact_desktop.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_desktop_test.parquet
# python inference/inference_vllm_omniact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_web_test.parquet
# python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_rede_test.parquet --num_actor 2 --prefix "<think>"
python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_rede_test.parquet --num_actor 2 # --prefix "<tool_call>"
python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_test.parquet --num_actor 2 # --prefix "<think>"
python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor 2 # --prefix "<think>"
python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_test.parquet --num_actor 2 --prefix "<think>"
python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor 2 --prefix "<think>"

# done