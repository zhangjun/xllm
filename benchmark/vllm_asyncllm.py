import json
import os
import asyncio
from transformers import AutoTokenizer
import torch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

# os.environ["VLLM_TORCH_PROFILER_DIR"] = './profile'
# Sample prompts.
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

async def main():
    model_path = os.environ.get("MODEL_PATH")
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        dtype="auto",
        kv_cache_dtype="fp8",
        max_model_len=131072,
        enforce_eager=False,
        # ray_workers_use_nsight=True,
        # distributed_executor_backend='ray',
    )

    engine_client = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=8,
    )

    messages = load_json("../write_model.json")[10]['messages']

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    prompt = prompt[:4096]

    final_output: Optional[RequestOutput] = None
    stream = engine_client.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id="abcdef",
    )
    async for output in stream:
        if output.finished:
            print(output.outputs[0].text)

if __name__ == "__main__":
    asyncio.run(main())
