MODEL_NAME=$1
DATA_DIR=/root/workspace/EasyR1/validation/guir1/outputs/$MODEL_NAME


python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_None.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_v2_test_None.json

python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_\<tool_call\>.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_v2_test_\<tool_call\>.json

python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_\<think\>.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_v2_test_\<think\>.json