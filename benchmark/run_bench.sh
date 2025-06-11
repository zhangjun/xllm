export HF_ENDPOINT=https://hf-mirror.com
export DATAPATH="/mnt/data/auto_bench/dataset/gemini-32k-64k.json"
export SERVE_MODEL_NAME=qw25_2050_agent_ge_15a_ep3_32kc_0517
export MODEL_PATH=/mnt/data/chat/${SERVE_MODEL_NAME}-FP8-Dynamic

concurrency_values=(4 8 12 16 20 24)
# request_rates=(128 64 32 16 8 4 2 1)

# for i in "${!request_rates[@]}";do
#     echo $i, ${max_concurrency[$i]}
# done

extra_params='{"temperature":"0.8","top_p":"1.0","stream_options":{"include_usage":true}}'

for concurrency in "${concurrency_values[@]}";do
    python3 bench_serving.py \
    --port 8000 \
    --backend openai \
    --model ${MODEL_PATH} \
    --tokenizer ${MODEL_PATH} \
    --model-id ${SERVE_MODEL_NAME} \
    --dataset-name custom \
    --dataset-path $DATAPATH \
    --num-prompt 120 \
    --use-chat-completions \
    --max-tokens 32768 \
    --max-concurrency "${concurrency}" \
    --request-rate 0.06 \
    --extra-request-body '{"temperature":"0.6","top_p":"1.0","stream_options":{"include_usage":true}}'
done
