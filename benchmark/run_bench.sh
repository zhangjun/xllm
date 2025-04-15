# export HF_ENDPOINT=https://hf-mirror.com
model_path=/mnt/data/llama3-8b
dataset_path=/mnt/data/vllm_sharegpt.jsonl

python sgl_bench_serving.py --backend vllm \
    --model $model_path \
    --dataset-path $dataset_path \
    --dataset-name random --num-prompts 128 \
    --random-input 2400 --random-output 32 \
    --random-range-ratio 0.5 \
    --request-rate 1 \
    --max-concurrency 6
