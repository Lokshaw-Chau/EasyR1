RESP_FILE=
EXP_NAME=$(basename $(dirname $RESP_FILE))
OUTPUT_DIR=/root/workspace/EasyR1/validation/agentnetbench/outputs/${EXP_NAME}

python reeval.py \
    --data_dir /root/workspace/EasyR1/data/gui-r1/AgentNetBench/test_data \
    --response_file ${RESP_FILE} \
    --agent_type qwen25vl \
    --output_dir ${OUTPUT_DIR}