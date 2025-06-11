# Python script
import json
import requests
from typing import Dict, List

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_model(model_url):
    response = requests.get(model_url)
    model_list = response.json().get("data", [])
    model = model_list[0]["id"] if model_list else None
    return model

#  <service_url>：替换为服务访问地址。
end_point = "http://localhost:8000"
use_chat = True
if use_chat:
    url = f"{end_point}/v1/chat/completions"
    prompt = load_json("../dataset/gemini-32k-64k.json")[10]['messages']
else:
    url = f"{end_point}/v1/completions"
    prompt = "hello world"

tools = load_json("tools_list.json")

model_url = f"{end_point}/v1/models"
model_id = get_model(model_url)
print(f"model_id: {model_id}")

req = {
    "model": model_id,
    "messages" if use_chat else "prompt": prompt,
    "tools": tools,
    "stream": True,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 10,
    "max_tokens": 512,
    "stream_options": {
        "include_usage": True,
    },    
}
print(f"req: {req}")
response = requests.post(
    url,
    json=req,
    # <Your EAS Token>：替换为服务Token。
    headers={"Content-Type": "application/json", "Authorization": "MDNiOGEwMTg4NTIyMWNmM2M0MzU4MjhhNjYwM2RhYjBlNWQxYWFmZQ=="},
    stream=True,
)
for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
    msg = chunk.decode("utf-8")
    print(msg)
    continue
    if msg.startswith('data'):
        info = msg[6:]
        if info == '[DONE]':
            break
        else:
            resp = json.loads(info)
            if "delta" in resp["choices"][0]:
                if "content" in resp["choices"][0]["delta"]:
                    print(resp["choices"][0]["delta"]["content"], end='', flush=True)
                else:
                    if "arguments" in resp["choices"][0]["delta"]["tool_calls"][0]["function"]:
                        print(resp["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], end='', flush=True)
                    else:
                        print(resp["choices"][0]["delta"]["tool_calls"][0]["function"]["name"])
            else:
                print(resp["choices"][0]["text"], end='', flush=True)