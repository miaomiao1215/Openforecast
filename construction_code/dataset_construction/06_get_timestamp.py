import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import copy
import argparse
from vllm import LLM, SamplingParams
import time
import os
import re
import numpy as np
from collections import defaultdict

def extract_dates(text):
    # 定义正则表达式
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',  # yyyy-MM-dd
        r'\b(\d{4}-\d{2})\b',         # yyyy-MM
        r'\b(\d{4})\b'                # yyyy
    ]
    
    # 提取所有匹配的日期
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
        if len(matches) > 0:
            break
    # 提取匹配结果中的第一个组（即完整匹配的日期）
    if len(dates) == 0:
        return None
    dates.sort(key=len, reverse=True)
    return dates[0]

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list, time_list = [], []

    for output in outputs:
        generated_text = output.outputs[0].text
        out_list.append(generated_text)
        time_list.append(extract_dates(generated_text))
    return out_list, time_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='data.json', help="do predict")
args = parser.parse_args()

prompt = "Please extract the occurrence time of the following event. Rules: \n1.The time format must follow \"yyyy\", \"yyyy-MM\", \"yyyy-MM-dd\", or \"Unknown\", where yyyy, MM, and dd refer to the year, month, and day respectively. \n2.Notice, the dates should be as precise as possible. If only the year is known, use the \"yyyy\" format. If the year and month are known, use the \"yyyy-MM\" format.\n3.If the event spans a certain period of time, please output the end time of the event.\n The event is: \"{event}\"\nThe occurrence time is:\n"

lines = open(args.dataset, 'r').readlines()

model_path = 'xx/llama3-8b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)

info_list, prompts_list, cluster_id_list = [], [], []
len_list = []
already_list = []
for line in lines:
    info = json.loads(line)

    article = ''
    info_list.append(info)

    prompt_temp = []

    for i, former_event_i in enumerate(info['event_list']):
        if info['source'] == 'wikipedia':
            prompt_i = copy.deepcopy(prompt).format(event=former_event_i)
        else:
            prompt_i = copy.deepcopy(prompt).format(event=former_event_i)
        messages = [{"role": "user", "content": prompt_i}]
        prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_temp.append(prompt_chat_i)

    prompts_list.append(prompt_temp)

# print(' average length: ', np.average(len_list))

if 'llama3' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=256, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=256, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 2048
find_num = 0
f_w = open(args.dataset.replace('.json', '_time.json'), 'w', encoding='utf-8')
for index in tqdm(range(0, len(prompts_list), batch_size)):
    prompts_list_sentecnces = []
    temp_j = 0
    index_dict = {}
    for i, prompts in enumerate(prompts_list[index: index+batch_size]):
        for prompt_j in prompts:
            prompts_list_sentecnces.append(prompt_j)
            index_dict[temp_j] = i
            temp_j += 1

    text_list, out_list = generate_vllm(llm, prompts_list_sentecnces, sampling_params)
    out_list_processed = defaultdict(list)
    for i, out_i in enumerate(out_list):
        out_list_processed[index_dict[i]].append(out_i)

    for i, info in enumerate(info_list[index: index+batch_size]):
        assert len(out_list_processed[i]) == len(info['event_list'])
        info['timestamp'] = out_list_processed[i]
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

for line in already_list:
    f_w.writelines(line)
f_w.close()

