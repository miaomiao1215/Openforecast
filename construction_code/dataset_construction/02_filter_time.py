import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
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
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer


def extract_dates(text):
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',  # yyyy-MM-dd
        r'\b(\d{4}-\d{2})\b',         # yyyy-MM
        r'\b(\d{4})\b'                # yyyy
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
        if len(matches) > 0:
            break

    if len(dates) == 0:
        return 'None'
    dates.sort(key=len, reverse=True)
    return dates[0]

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []

    for output in outputs:
        generated_text = output.outputs[0].text
        time = extract_dates(generated_text)
        out_list.append(time)
    return out_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='./wikipedia/wikipedia_en_20240320_filter.json', help="do predict")

args = parser.parse_args()
prompt = "Please extract the occurrence time of the following event. Rules: \n1.The time format must follow \"yyyy\", \"yyyy-MM\", or \"yyyy-MM-dd\", where yyyy, MM, and dd refer to the year, month, and day respectively. \n2.Notice, the dates should be as precise as possible. If only the year is known, use the \"yyyy\" format. If the year and month are known, use the \"yyyy-MM\" format.\n3.If the event spans a certain period of time, please output the end time of the event.\n The wikipedia title: {title}\n The article: {article}\nThe occurrence time is:\n"

lines = open(args.dataset, 'r').readlines()
model_path = './llama3-8b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)

info_list, prompts_list, cluster_id_list = [], [], []
len_list = []
for line in lines:
    info = json.loads(line)
    article = ''
    info_list.append(info)
    for section, text in zip(info['section_titles'], info['section_texts']):
        if section.strip() in ['Introduction', 'Results', 'Aftermath', 'Events', 'Overview', 'Summary', 'Reactions', 'Battle', \
                               'Investigation', 'Impact', 'Timeline', 'Event', 'Incident', 'Accident', 'Reaction', 'Development', 'Response', 'Effects', \
                                'Description', 'Causes', 'Cause', 'Storylines', 'Incidents', 'Consequences', 'Responses', 'Protests', 'Massacre', \
                                'Investigations', 'Outcome', 'Attacks']:
            article += '== %s ==\n %s \n'%(section.strip(), text.strip())
        if article.count(' ') > 5000:
            break
    len_list.append(len(article))
    prompt_i = copy.deepcopy(prompt_en).format(title=info['title'], article=article)
    messages = [{"role": "user", "content": prompt_i}]
    prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts_list.append(prompt_chat_i)

print(' average length: ', np.average(len_list))
print('select number:', len(info_list))
assert len(info_list) == len(prompts_list)
sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=128, n=1)
llm = LLM(model="xx/llama3-8b-instruct", tensor_parallel_size=2, quantization=None, dtype='bfloat16')

batch_size = 2048
out_list, result_list = [], []
find_num = 0
f_w = open(args.dataset.replace('.json', '_time.json'), 'w', encoding='utf-8')
for index in tqdm(range(0, len(prompts_list), batch_size)):
    out_list_i = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)

    for info, out in zip(info_list[index: index+batch_size], out_list_i):
        info['year'] = out.strip()
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

    print('find number: ', find_num)
f_w.close()

