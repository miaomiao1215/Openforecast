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

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []
    # pattern = re.compile(r'Yes', re.IGNORECASE)
    pattern = re.compile(r'Event', re.IGNORECASE)

    for output in outputs:
        generated_text = output.outputs[0].text
        out_list.append(generated_text)
    return out_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='./wikipedia/wikipedia_en_20240320_filter.json', help="do predict")

args = parser.parse_args()
prompt_1 = "Please extract structured event information from the given event text. The rules are as follows: \n1. Structured event information includes event trigger words and parameters such as time, location, subject, and object. If the above parameters are not enough to express the event clearly, you can also use other elements such as announcement content, condemnation content, boycott content and other elements (excluding causes and consequences); \n 2. Event elements such as subject and object should use full names instead of Pronouns (such as he, she, they, etc.);\n3. Event text can contain multiple structured events. Please output events in chronological order. If there are no structured events, output \"None\". The format is as shown in the example. \n An example is as follows: \nEvent text: \n In 2014, Russian troops occupied Crimea, and Russia quickly annexed it. Putin stated that he would establish a Russian military task force in Crimea\nThe extracted structured event list is: [{Event: occupation; Time: 2014; Location: Crimea; Subject: Russian troops; Object: Crimea}, {event: annexation; time: 2014; subject: Russia; object: Crimea}, {event: make statement; time: 2014; subject: Putin; statement content: he would establish a Russian military task force in Crimea}]\n\n Now, please perform event extraction on the following event text. Event text:\n%s\nThe extracted structured event list is:\n"

lines = open(args.dataset, 'r').readlines()

model_path = "xx/llama3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

info_list, prompts_list, cluster_id_list = [], [], []
len_list = []
for line in lines:
    info = json.loads(line)
    article = ''
    info_list.append(info)

    prompt_temp = []
    for i, former_event_i in enumerate(info['event_list']):
        prompt_i = copy.deepcopy(prompt_1)%former_event_i
        messages = [{"role": "user", "content": prompt_i}]
        prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_temp.append(prompt_chat_i)


    prompts_list.append(prompt_temp)


if 'llama3' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 512
find_num = 0
f_w = open('./wikipedia/wikipedia_en_20240320_filter.json', 'w', encoding='utf-8')
for index in tqdm(range(0, len(prompts_list), batch_size)):
    prompts_list_sentecnces = []
    temp_j = 0
    index_dict = {}
    for i, prompts in enumerate(prompts_list[index: index+batch_size]):
        for prompt_j in prompts:
            prompts_list_sentecnces.append(prompt_j)
            index_dict[temp_j] = i
            temp_j += 1

    out_list = generate_vllm(llm, prompts_list_sentecnces, sampling_params)
    out_list_processed = defaultdict(list)
    for i, out_i in enumerate(out_list):
        out_list_processed[index_dict[i]].append(out_i)

    for i, info in enumerate(info_list[index: index+batch_size]):
        assert len(out_list_processed[i]) == len(info['event_list'])
        info['event_list_extacted'] = out_list_processed[i]
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

f_w.close()

