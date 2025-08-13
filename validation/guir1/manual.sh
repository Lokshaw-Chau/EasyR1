
export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_ACTOR=4
cd /root/workspace/EasyR1

SAVE_PATH=/root/workspace/EasyR1/ckpts
EXP_NAME=qwen2_5_vl_3b_adaptive_PRI_BCP0.2_TP0.2_sepGAE_modeKL0.1-0_NH

python scripts/model_merger.py --local_dir $SAVE_PATH/$EXP_NAME/global_step_69/actor

mv $SAVE_PATH/$EXP_NAME/global_step_69/actor/huggingface $SAVE_PATH/$EXP_NAME/global_step_69/actor/$EXP_NAME-69steps

cd /root/workspace/EasyR1/validation/guir1

exps=($SAVE_PATH/$EXP_NAME/global_step_69/actor/$EXP_NAME-69steps)

DATA_DIR=/root/workspace/EasyR1/data
# TEST_FILE_DELIB=${DATA_DIR}/uground_more_bias_sampled_delib_256.parquet
# TEST_FILE_REACT=${DATA_DIR}/uground_more_bias_sampled_react_256.parquet


for MODEL_PATH in ${exps[@]}
do

    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor ${NUM_ACTOR} 
    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_v2_test.parquet --num_actor ${NUM_ACTOR} 

    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor ${NUM_ACTOR}  --prefix "<think>"
    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_v2_test.parquet --num_actor ${NUM_ACTOR}  --prefix "<think>"

    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet --num_actor ${NUM_ACTOR} --prefix "<tool_call>"
    python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_v2_test.parquet --num_actor ${NUM_ACTOR} --prefix "<tool_call>"

    # python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path $TEST_FILE_DELIB --num_actor ${NUM_ACTOR}
    # python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path $TEST_FILE_REACT --num_actor ${NUM_ACTOR}

    # python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path $TEST_FILE_DELIB --num_actor ${NUM_ACTOR} --prefix "<think>"
    # python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path $TEST_FILE_REACT --num_actor ${NUM_ACTOR} --prefix "<think>"

    # python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path $TEST_FILE_DELIB --num_actor ${NUM_ACTOR} --prefix "<tool_call>"
    # python inference/inference_vllm_screenspot_official.py --model_path ${MODEL_PATH} --data_path $TEST_FILE_REACT --num_actor ${NUM_ACTOR} --prefix "<tool_call>"

done


DATA_DIR=/root/workspace/EasyR1/validation/guir1/outputs/$EXP_NAME-69steps
MODEL_NAME=$EXP_NAME-69steps

python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_None.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_v2_test_None.json

python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_\<tool_call\>.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_v2_test_\<tool_call\>.json

python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_\<think\>.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_v2_test_\<think\>.json