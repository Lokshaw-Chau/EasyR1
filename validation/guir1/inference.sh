
export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_ACTOR=4

exps=($1)

DATA_DIR=/root/workspace/EasyR1/data

for MODEL_PATH in ${exps[@]}
do

    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor ${NUM_ACTOR} 
    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_v2_test.parquet --num_actor ${NUM_ACTOR} 

    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor ${NUM_ACTOR}  --prefix "<think>"
    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_v2_test.parquet --num_actor ${NUM_ACTOR}  --prefix "<think>"

    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor ${NUM_ACTOR} --prefix "<tool_call>"
    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_v2_test.parquet --num_actor ${NUM_ACTOR} --prefix "<tool_call>"

done

