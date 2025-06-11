import pandas as pd
import json
import numpy as np

# 新增行：df.loc[row_index] = [val1, val2, val3]
# 新增列：df[colo_name] = None

# df = pd.read_excel("s.xlsx")
# print(df.columns)

columns = ['发送 QPS', '实测 QPS', '并发上限', 'Avg Throughput (tokens/s)',
       'Output tokens/s', 'TTFT@avg (s)', 'TPOT@avg (s)', 'Latency@avg (s)',
       'Decode Speed tokens/s', 'input avg', 'input range [min,max]:std',
       'output avg', 'output range [min,max]:std']
df = pd.DataFrame({},columns=columns)
df.to_excel("test.xlsx", sheet_name='stat', index=False, header=True)

data = []
with open('benchmark_results_061019.jsonl') as fp:
    for line in fp:
        data.append(json.loads(line))

# print(data[0])
for idx, item in enumerate(data):
    value = [
        item['request_rate'],
        round(item['request_throughput'], 2),
        item['max_concurrency'],
        round(item['output_throughput'] + item['input_throughput'], 2),
        round(item['output_throughput'], 2),
        round(item['mean_ttft_ms'] * 0.001, 2),
        round(item['mean_tpot_ms'] * 0.001, 2),
        round(item['mean_e2e_latency_ms'] * 0.001, 2),
        round(1/item['mean_tpot_ms'] * 1000, 3),
        round(np.mean(item['input_lens'] or 0), 2),
        f"[{np.min(item['input_lens'] or 0)},{np.max(item['input_lens'] or 0)}]:{np.std(item['input_lens'] or 0)}",
        round(np.mean(item['output_lens'] or 0), 2),
        f"[{np.min(item['output_lens'] or 0)},{np.max(item['output_lens'] or 0)}]:{np.std(item['output_lens'] or 0)}",
    ]
    df.loc[idx + 1] = value

df.to_excel("test.xlsx", sheet_name='stat', index=False, header=True)
