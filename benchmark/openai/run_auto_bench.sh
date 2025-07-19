export HF_ENDPOINT=https://hf-mirror.com

DATASET="gemini-32k-64k.json" # [{"messages":""},...,{"messages":""}]
TOOLS_PATH=tools_list.json.   # tool params

SERVE_MODEL_NAME=qw3_2050_agent_merge_planner5_0704
MODEL_PATH=/mnt/shared/maas/2050/chat/${SERVE_MODEL_NAME}-fp8
export model_id=$SERVE_MODEL_NAME

export openai_url='http://localhost:8000/v1/completions'
export openai_url='https://api-maas-gateway.singularity-ai.com/sky-chat-saas/api/v1/chat/completions'
export OPENAI_API_KEY='ccd25bb89432fd32e0ff6c4839187797'


QPS=(0.1 0.2 1.8 2.0 2.2 2.4)

for qps in "${QPS[@]}";do

    TASKNAME=vllm_${model_id}
    # --tools ${TOOLS_PATH} \

    python3 auto_bench.py \
        --task-name ${TASKNAME} \
        --model $model_id \
        --intent chat \
        --data-path ${DATASET} \
        --tokenizer ${MODEL_PATH} \
        --time-size 12 \
        --chat-completions \
        --openai-url ${openai_url} \
        --qps ${qps} \
        --llm-params "{\"n\":\"1\",\"temperature\":\"0\",\"top_p\":\"1.0\",\"max_tokens\":\"4096\",\"length_penalty\":\"0.5\"}"

done
