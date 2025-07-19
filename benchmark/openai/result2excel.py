import pandas as pd
import json
import numpy as np

label = ['发送 QPS', '实测 QPS', '并发上限', 'Avg Throughput (tokens/s)', 
         'Output tokens/s', 'TTFT@avg (s)', 'TPOT@avg (s)', 'Latency@avg (s)', 
         'Decode Speed tokens/s', 'input avg', 'input range [min,max]:std', 
         'output avg', 'output range [min,max]:std]']

df = pd.DataFrame({}, columns=label)
df.to_excel("result.xlsx", sheet_name='stat', index=False, header=True)

data = []
with open('input.json') as fp:
    for line in fp:
        data.append(json.loads(line))

# print(data[0])
for idx, item in enumerate(data):
    value = [
        '',
        item['requests_per_second'],
        item['max_concurrent'],
        item['tokens_per_second'],
        item['output_tokens_per_second'],
        item['ttft@avg'],
        item['tpot@avg'],
        item['latency@avg'],
        round(1/item['tpot@avg'], 3),
        item['prompt_len@avg'],
        f"[{item['prompt_len@min']},{item['prompt_len@max']}]:{item['prompt_len@std']}",
        item['output_len@avg'],
        f"[{item['output_len@min']},{item['output_len@max']}]:{item['output_len@std']}",
    ]
    df.loc[idx + 1] = value

df.to_excel("result.xlsx", sheet_name='stat', index=False, header=True)
