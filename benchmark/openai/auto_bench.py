import argparse
from openai import AsyncOpenAI
import asyncio
import datetime
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import aiohttp
import numpy as np
import requests
try:
    from loguru import logger
except:
    print("loguru not installed, using default logger")
    import logging
    logger = logging.getLogger(__name__)
from requests import RequestException
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@dataclass
class Response:
    prompt: str = field(default="")
    generated: str = field(default="")
    prompt_len: int = field(default=0)
    output_len: int = field(default=0)
    latency: float = field(default=0.0)
    ttft: Optional[float] = field(default=None)
    tpot: Optional[float] = field(default=None)
    sky_trace_id: Optional[str] = field(default=None)


REQUEST_RESPONSES: List[Response] = []
PERCENTILES = [25, 50, 75, 90, 95, 99, 99.9, 99.99]


def round_if_float(value, decimals):
    return round(value, decimals) if isinstance(value, float) else int(value)


def statistics_math_attribute(data, label, phase, result_data):
    avg = round_if_float(np.mean(data), 3)
    std = round_if_float(np.std(data), 3)
    min_val = round_if_float(np.min(data), 3)
    max_val = round_if_float(np.max(data), 3)

    logger.info(f"{phase} Average {label}: {avg}")
    logger.info(f"{phase} Standard {label}: {std}")
    logger.info(f"{phase} Minimum {label}: {min_val}")
    logger.info(f"{phase} Maximum {label}: {max_val}")

    result_data[f"{label}@avg"] = avg
    result_data[f"{label}@std"] = std
    result_data[f"{label}@min"] = min_val
    result_data[f"{label}@max"] = max_val


async def send_request(
    messages: list,
    prompt: str,
    prompt_len: int,
    llm_params: dict,
    model: str,
    tokenizer: PreTrainedTokenizerFast,
    infer_env: str,
    openai_url,
) -> None:
    request_start_time = time.perf_counter()

    features = {"chatsaas::prompt_msgs": True}
    app_key = "empty"
    app_secret = "empty"

    app_url = openai_url or app_url
    headers = {
        "app_key": app_key,
        "Content-Type": "application/json",
        "stream": "true",
    }

    pload = {
        "messages": messages,
        "integrated_params": {
            "conversation_id": uuid.uuid4().hex,
            "ask_id": uuid.uuid4().hex,
            "device_id": "",
            "device_hash": "",
            "app_version": "",
            "user_id": "",
            "request_id": uuid.uuid4().hex,
            "ask_type": "",
            "ask_from": "",
            "device": "android",
            "product_name": "auto-llm-eval",
            "fixed_image": False,
            "agent_id": "016",
            "source": "auto-llm-eval",
        },
        "intent_returned": True,
        "ab_params": "",
        "features": features,
        "model_params": {"chat": model},
        "model": model,
        "intent": 'chat',
        "use_agent": False,
        "llm_params": llm_params or {}
    }
    if openai_url:
        pload = {
            "stream": True,
            "model": model,
        }
        if args.chat_completions:
            pload["messages"] = messages
            if "tools" in llm_params and llm_params["tools"] is not None:
                pload["tools"] = llm_params["tools"]
        else:
            pload["prompt"] = prompt
        if 'generate_length' in llm_params:
            pload['max_tokens'] = llm_params['generate_length']
        if 'max_tokens' in llm_params:
            pload['max_tokens'] = llm_params['max_tokens']
        if 'top_p' in llm_params:
            pload['top_p'] = llm_params['top_p']
        if 'top_k' in llm_params:
            pload['top_k'] = llm_params['top_k']
        if 'temperature' in llm_params:
            pload['temperature'] = llm_params['temperature']
        if 'repetition_penalty' in llm_params:
            pload['repetition_penalty'] = llm_params['repetition_penalty']
        if 'frequency_penalty' in llm_params:
            pload['frequency_penalty'] = llm_params['frequency_penalty']
        if 'end_words' in llm_params:
            pload['stop'] = llm_params['end_words']
        if 'stop' in llm_params:
            pload['stop'] = llm_params['stop']

    timeout = aiohttp.ClientTimeout(total=3600 * 3)
    MAX_RETRIES = 2
    sky_trace_id = None
    ttft = None
    tpot = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(MAX_RETRIES):
            chunk_cursor = ['']
            try:
                timestamp = str(int(time.time()))
                sign_content = app_key + app_secret + timestamp
                sign_result = hashlib.md5(sign_content.encode('utf-8')).hexdigest()
                sky_trace_id = f"{uuid.uuid4().hex}:{uuid.uuid4().hex[:16]}:{'0' * 16}:1"

                headers["timestamp"] = timestamp
                headers["sign"] = sign_result
                headers["sky_trace_id"] = sky_trace_id

                if openai_url:
                    prompt_len, output_len, output, tpot_interval, ttft, tpot = await extract_siliconflow_output(
                        session,
                        app_url,
                        headers,
                        pload,
                        request_start_time,
                        chunk_cursor,
                        ttft,
                        tpot,
                    )
                else:
                    output, tpot_interval, ttft, tpot = await extract_maas_output(
                        session,
                        app_url,
                        headers,
                        pload,
                        request_start_time,
                        chunk_cursor,
                        ttft,
                        tpot,
                    )

                output_len = len(tokenizer.tokenize(output))
                if output_len > 1:
                    tpot = tpot_interval / (output_len - 1)
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                if isinstance(e, UnicodeDecodeError):
                    logger.error(f"UnicodeDecodeError: {e}, chunk: {chunk_cursor[0]}")
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, pload: {str(pload)}, headers: {str(headers)}, retrying")
                else:
                    raise Exception(f"All {MAX_RETRIES} attempts failed: {e}, pload: {str(pload)}, headers: {str(headers)}")

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time

    global REQUEST_RESPONSES
    REQUEST_RESPONSES.append(
        Response(
            prompt= prompt if openai_url and not args.chat_completions else json.dumps(messages, ensure_ascii=False),
            generated=output,
            prompt_len=prompt_len,
            output_len=output_len,
            latency=request_latency,
            ttft=ttft,
            tpot=tpot,
            sky_trace_id=sky_trace_id,
        )
    )


async def extract_maas_output(
    session,
    app_url,
    headers,
    pload,
    request_start_time,
    chunk_cursor,
    ttft,
    tpot,
):
    async with session.post(
            app_url, headers=headers, json=pload
    ) as response:
        chunks_text = ''
        async for chunk, _ in response.content.iter_chunks():
            chunk_cursor[0] = chunk
            if ttft is None:
                first_token_time = time.perf_counter()
                ttft = first_token_time - request_start_time
            chunks_text += chunk.rstrip(b"\x00").decode("utf-8")

    tpot_interval = time.perf_counter() - first_token_time

    outputs = []
    lines = chunks_text.split('\n\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('data: '):
            line = line[6:]
        if line == '[DONE]':
            break

        try:
            obj = json.loads(line)
            if obj.get('card_type', '') == 'markdown':
                text = obj['arguments'][0]['messages'][0]['text']
                outputs.append(text)
            if 'code' in obj and str(obj.get('code', '')) != '200':
                code = obj.get('code', '')
                code_msg = obj.get('code_msg', '')
                raise RequestException(f'RPC failed: {code}, {code_msg}')
        except Exception as err:
            raise Exception(f"Failed to parse response: {err}, line: {line}")
    output = "".join(outputs)
    return output, tpot_interval, ttft, tpot


openai_client = None

async def extract_siliconflow_output(
    session,
    app_url,
    headers,
    pload,
    request_start_time,
    chunk_cursor,
    ttft,
    tpot,
):
    global openai_client
    if openai_client is None:
        assert "/v1" in app_url, app_url
        openai_base = app_url.split("/v1")[0] + '/v1'
        openai_client = AsyncOpenAI(base_url=openai_base, api_key=os.environ.get("OPENAI_API_KEY", "no_key"))

    pload = pload.copy()
    model = pload.pop("model")
    assert pload.pop("stream", False)
    try:
        if args.chat_completions:
            messages = pload.pop("messages")
            stream_options={"include_usage": True, "continuous_usage_stats": True}
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                extra_body=pload,
                stream_options=stream_options,
            )
        else:
            prompt = pload.pop("prompt")
            stream = await openai_client.completions.create(
                model=model,
                prompt=prompt,
                stream=True,
                extra_body=pload,
            )
    except Exception as e:
        logger.error(f"创建stream失败: {e}")
        raise
        
    chunks = []
    i = 0
    first_token_time = None
    prompt_tokens, completion_tokens = 0, 0
    async for chunk in stream:
        i += 1
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        if len(chunk.choices) == 0:
            continue
        if args.chat_completions:
            if chunk.choices[0].delta.tool_calls:
                text = chunk.choices[0].delta.tool_calls[0].function.arguments
            else:
                text = chunk.choices[0].delta.content
        else:
            text = chunk.choices[0].text
        if text is None or len(text) == 0:
            continue
        if ttft is None:
            first_token_time = time.perf_counter()
            ttft = first_token_time - request_start_time
            if i != 0:
                print(f'第 {i} 个 token 才非空')
        chunks.append(chunk)
    
    if first_token_time is None:
        if args.chat_completions:
            print(f'未收到有效结果，跳过，{messages=}, {pload=}')
        else:
            print(f'未收到有效结果，跳过，{prompt=}, {pload=}')
        return 0, 0, '', 0, 0, 0

    tpot_interval = time.perf_counter() - first_token_time

    outputs = []
    for chunk in chunks:
        chunk_cursor[0] = chunk
        try:
        #     data = chunk.rstrip(b"\x00").lstrip(
        #         b"data:").rstrip(b"\n\n").strip().decode(
        #         "utf-8")
        #     output = json.loads(data)
        #     text = output["choices"][0]['text']
        #     outputs.append(text)
        # except json.decoder.JSONDecodeError as err:
        #     logger.warning(f"{err}, data: {data}")
        #     continue
            if args.chat_completions:
                if chunk.choices[0].delta.tool_calls:
                    text = chunk.choices[0].delta.tool_calls[0].function.arguments
                else:
                    text = chunk.choices[0].delta.content
                if text is None:
                    text = ''
            else:
                text = chunk.choices[0].text
            outputs.append(text)
        except Exception as err:
            raise Exception(f"Failed to parse response: {err}, chunk: {chunk}")
    output = "".join(outputs)
    return prompt_tokens, completion_tokens, output, tpot_interval, ttft, tpot


async def benchmark(
    input_requests: List[Tuple[list, str, int, dict]],
    model: str,
    tokenizer: PreTrainedTokenizerFast,
    infer_env: str,
    openai_url: str,
    time_size: int,
    qps: float,
) -> Tuple[float, float, int, int]:
    async_lock = asyncio.Lock()
    tasks: List[asyncio.Task] = []
    query_intervals = 1 / qps
    wait_time = 0
    call_cnt = 0
    done_cnt = [0]
    expect_call_times = min(time_size * 60 * qps, len(input_requests))
    progress_bar = async_tqdm(
        total=expect_call_times,
        desc="Processing Requests",
        smoothing=0.0,
    )
    max_concurrent = 0
    start_time = time.perf_counter()
    for request in input_requests:
        if time.perf_counter() - start_time >= time_size * 60:
            break
        messages, prompt, prompt_len, llm_params = request
        task = asyncio.create_task(
            send_request(
                messages,
                prompt,
                prompt_len,
                llm_params,
                model,
                tokenizer,
                infer_env,
                openai_url,
            )
        )
        call_cnt += 1
        max_concurrent = max(max_concurrent, call_cnt - done_cnt[0])
        task.add_done_callback(
            lambda _: asyncio.create_task(
                query_done_callback(progress_bar, done_cnt, async_lock)
            )
        )
        tasks.append(task)
        await asyncio.sleep(query_intervals)
        wait_time += query_intervals
    logger.info(
        f"Expected number of calls: {expect_call_times},"
        f"practically call {call_cnt} times,"
        f"spend {time.perf_counter() - start_time} second."
    )
    await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    progress_bar.close()
    duration = (end_time - start_time)
    duration_without_wait = duration - wait_time
    return duration, duration_without_wait, max_concurrent, call_cnt


async def query_done_callback(progress_bar, done_cnt, async_lock):
    async with async_lock:
        done_cnt[0] += 1
        progress_bar.update()


def sample_policy(dataset, tokenizer, llm_params, openai_url):
    sampled_dataset = []
    print(f'{args.chat_completions=}')
    for sample in dataset:
        if openai_url and not args.chat_completions:
            messages = []
            prompt = sample.get("prompt")
            if not prompt:
                continue
        else:
            messages = sample.get("messages")
            if isinstance(messages, str):
                messages = json.loads(messages)
            if any(not message.get('content') for message in messages):
                continue

            prompt = ''
            for prompts in messages:
                if prompts['role'] == 'bot':
                    print(f'将 {prompts} 的 role 从 bot 改为 assistant')
                    prompts['role'] = 'assistant'
                prompt += prompts['content']

        tokens = tokenizer.tokenize(prompt)
        token_len = len(tokens)

        llm_params = llm_params or (json.loads(sample["param_send"]) if 'param_send' in sample else {})
        
        # enable tool
        if "tools" in llm_params:
            llm_params["tools"] = sample.get("tools") if llm_params["tools"] == "auto" else llm_params["tools"]
        sampled_dataset.append((messages, prompt, token_len, llm_params))

    return sampled_dataset


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--intent", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=False, help='output filename prefix, default to model name')
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--tools", type=str, required=False, help='tool list path or load from dataset')
    parser.add_argument("--tokenizer", type=str, default='Qwen/Qwen2-72B-Instruct')
    parser.add_argument("--ramp-up-period", type=int, default=60)
    parser.add_argument("--openai-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--time-size", type=int)
    parser.add_argument("--qps", type=float)
    parser.add_argument("--llm-params", type=str)
    parser.add_argument("--chat-completions", action="store_true")
    args = parser.parse_args()
    logger.info(f"params: {args}")

    intent = args.intent
    model = args.model
    data_path = args.data_path
    openai_url = args.openai_url
    time_size = args.time_size
    qps = args.qps
    expect_call_times = time_size * 60 * qps
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer)
    ramp_up_period = args.ramp_up_period
    llm_params = json.loads(args.llm_params) if args.llm_params else {}
    task_name = args.task_name or model
    if '/' not in task_name:
        pred_base_path = task_name
    else:
        pred_base_path = os.path.basename(task_name.rstrip('/')) or os.path.basename(os.path.dirname(task_name))

    pred_path = f"{pred_base_path}.jsonl"
    statistics_path = f"{pred_base_path}-statistics.json"

    # dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.data_shuffle:    
        import random
        random.shuffle(dataset)

    if args.tools:
        llm_params['tools'] = load_json(args.tools) if os.path.exists(args.tools) else args.tools

    # for item in load_data:
    # for line in f:
    #     dataset.append({"message": json.loads(line)})

    infer_env = 'test'
    sample_requests = sample_policy(dataset, tokenizer, llm_params, openai_url)
    num_warmup_requests = int(max(qps, 0.25) * 4)
    sampled_requests_for_warmup = sample_requests[:num_warmup_requests]
    sampled_requests_for_benchmark = sample_requests[num_warmup_requests:]

    REQUEST_RESPONSES = []
    for phase, input_requests in zip(
        ("Warmup", "Benchmark"),
        (sampled_requests_for_warmup, sampled_requests_for_benchmark),
    ):
        duration, duration_without_wait, max_concurrent, real_call_num = (
            asyncio.run(
                benchmark(
                    input_requests,
                    model,
                    tokenizer,
                    infer_env,
                    openai_url,
                    time_size,
                    qps,
                )
            )
        )
        if phase == "Benchmark":
            REQUEST_RESPONSES = REQUEST_RESPONSES[num_warmup_requests:]
        else:
            continue  # skip warmup phase

        for response in REQUEST_RESPONSES:
            record = asdict(response)
            with open(pred_path, "a", encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        result_data = OrderedDict(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "model": model,
                "num_warmup_requests": num_warmup_requests,
                "num_benchmark_requests": real_call_num,
                "max_concurrent": max_concurrent,
            }
        )
        if phase == "Benchmark":
            result_data["num_benchmark_requests"] = real_call_num
        logger.info(f"{phase} Total time: {duration:.2f} s")
        logger.info(f"{phase} Total time excluding waiting time: {duration_without_wait:.2f} s")
        result_data["total_time"] = round(duration, 2)
        total_tokens = np.sum(
            [
                (response.prompt_len + response.output_len)
                for response in REQUEST_RESPONSES
            ]
        )
        generate_tokens = np.sum(
            [response.output_len for response in REQUEST_RESPONSES]
        )
        logger.info(
            f"{phase} Throughput: {real_call_num / duration:.2f} requests/s, {total_tokens / duration:.2f} tokens/s, {generate_tokens / duration:.2f} tokens/s"
        )
        result_data["tokens_per_second"] = int(total_tokens / duration)
        result_data["output_tokens_per_second"] = int(generate_tokens / duration)
        result_data["requests_per_second"] = round(real_call_num / duration, 2)

        # Compute the latency statistics.
        latencies = [response.latency for response in REQUEST_RESPONSES]
        statistics_math_attribute(latencies, "latency", phase, result_data)

        latency_percentile = ", ".join(
            [
                f"P{k} = {v:.3f}"
                for k, v in
                zip(PERCENTILES, np.percentile(latencies, PERCENTILES))
            ]
        )
        logger.info(
            f"{phase} Latency: avg = {np.mean(latencies):.3f}, {latency_percentile}"
        )
        for percentile, v in zip(PERCENTILES, np.percentile(latencies, PERCENTILES)):
            result_data[f"latency@P{percentile}"] = round(v, 3)

        avg_per_token_latency = np.mean(
            [
                response.latency / (response.prompt_len + response.output_len)
                for response in REQUEST_RESPONSES
            ]
        )
        logger.info(f"{phase} Average latency per token: {avg_per_token_latency:.2f} s")
        result_data["avg_latency_per_prompt_token"] = round(avg_per_token_latency, 3)
        avg_per_output_token_latency = np.mean(
            [
                response.latency / response.output_len
                for response in REQUEST_RESPONSES
                if response.output_len
            ]
        )
        logger.info(
            f"{phase} Average latency per output token: "
            f"{avg_per_output_token_latency:.2f} s"
        )
        result_data["avg_latency_per_output_token"] = round(
            avg_per_output_token_latency, 3
        )

        ttft = [response.ttft or np.nan for response in REQUEST_RESPONSES]
        ttft_percentile = ", ".join(
            [
                f"P{k} = {v:.3f}"
                for k, v in
                zip(PERCENTILES, np.nanpercentile(ttft, PERCENTILES))
            ]
        )
        logger.info(f"{phase} TTFT: avg = {np.nanmean(ttft):.3f}, {ttft_percentile}")
        result_data["TTFT@Avg"] = round(np.nanmean(ttft), 3)
        for percentile, v in zip(PERCENTILES, np.nanpercentile(ttft, PERCENTILES)):
            result_data[f"TTFT@P{percentile}"] = round(v, 3)
        tpot = [response.tpot or np.nan for response in REQUEST_RESPONSES]
        tpot_percentile = ", ".join(
            [
                f"P{k} = {v:.3f}"
                for k, v in
                zip(PERCENTILES, np.nanpercentile(tpot, PERCENTILES))
            ]
        )
        logger.info(f"{phase} TPOT: avg = {np.nanmean(tpot):.3f}, {tpot_percentile}")
        result_data["TPOT@Avg"] = round(np.nanmean(tpot), 3)
        for percentile, v in zip(PERCENTILES, np.nanpercentile(tpot, PERCENTILES)):
            result_data[f"TPOT@P{percentile}"] = round(v, 3)

        prompt_lens = [response.prompt_len for response in REQUEST_RESPONSES]
        statistics_math_attribute(prompt_lens, "prompt_len", phase, result_data)

        output_lens = [response.output_len for response in REQUEST_RESPONSES]
        statistics_math_attribute(output_lens, "output_len", phase, result_data)
        result_data["output_len_0"] = sum(1 for response in REQUEST_RESPONSES if response.output_len == 0)

        result_data["intent"] = intent
        result_data["ramp_up_period"] = ramp_up_period

        new_data = OrderedDict()
        for k, v in result_data.items():
            k = k.lower()
            new_data[k] = v

        with open(statistics_path, "a", encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False)
            f.write("\n")
