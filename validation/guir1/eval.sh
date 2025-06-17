
DATA_DIR=/root/workspace/EasyR1/validation/guir1/outputs/huggingface
MODEL_NAME=qwen2_5_vl_3b_0615-150steps

# python evaluation/eval_omni.py --model_id ${MODEL_NAME} --prediction_file_path  ${DATA_DIR}/androidcontrol_high_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/androidcontrol_low_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/guiact_web_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/guiodyssey_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/omniact_desktop_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/omniact_web_test.json
# python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_None.json
# python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_test_None.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_rede_test_None.json

# python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_rede_test_\<tool_call\>.json
# python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test_\<think\>.json
# python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_test_\<think\>.json