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

def extract_event(output):
    event_list = []
    for out_i in output.outputs:
        text_i = out_i.text
        event_list = []
        if '1.' not in text_i:
            continue
        for j in range(2, 100):
            if '%i.'%j in text_i:
                index_start = text_i.index('%i.'%(j-1)) + len('%i.'%j)
                index_end = text_i.index('%i.'%j)
                event_list.append(text_i[index_start: index_end].strip())
            else:
                index_start = text_i.index('%i.'%(j-1)) + len('%i.'%j)
                event_list.append(text_i[index_start:].strip().split('\n')[0])
                break
        event_list_filter = []
        for event_i in event_list:
            if len(event_i) > 0:
                event_list_filter.append(event_i)
        if len(event_list_filter) > 0:
            return event_list_filter
    return event_list

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []
    # pattern = re.compile(r'Yes', re.IGNORECASE)
    pattern = re.compile(r'Event', re.IGNORECASE)

    for output in outputs:
        generated_text = output.outputs[0].text
        event_list = extract_event(output)
        out_list.append(event_list)
    return out_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='./wikipedia/wikipedia_en_20240320_filter.json', help="do predict")

args = parser.parse_args()

prompt_1 = "Please adjust the sequence of events according to the given article to correct any chronological errors in given event tineline, ensuring that the events are arranged in chronological order \n The article title: {title}\n The article: {article}\n The original event timeline: {timeline}\n The adjusted event timeline is: "

lines = open(args.dataset, 'r').readlines()[8191:]

model_path = "xx/llama3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

info_list, prompts_list, cluster_id_list, input_ids_list, prompts_chat_list = [], [], [], [], []
len_list = []
for line in tqdm(lines):
    info = json.loads(line)
    article = ''
    info_list.append(info)
    for section, text in zip(info['section_titles'], info['section_texts']):
        if section.strip() in ['Introduction', 'Background', 'Results', 'Aftermath', 'Events', 'Overview', 'Summary', 'Reactions', \
                               'Investigation', 'Impact', 'Timeline', 'Event', 'Incident', 'Accident', 'Reaction', 'Development', 'Response', 'Effects', \
                                'Description', 'Causes', 'Cause', 'Storylines', 'Incidents', 'Consequences', 'Responses', 'Protests', 'Massacre', \
                                'Outcome', 'Attacks']:
            article += '== %s ==\n %s \n'%(section.strip(), text.strip())
    len_list.append(len(article))
    timeline = ''
    for i, former_event_i in enumerate(info['event_list']):
        timeline += '%i.%s\n'%(i+1, former_event_i)
    # if len(article) > 5000:
    #     article = article[0: 5000] + '...'
    prompt_i = copy.deepcopy(prompt_1).format(title=info['title'], article=article, timeline=timeline)
    messages = [{"role": "user", "content": prompt_i}]
    prompts_list.append(prompt_i)

print(' average length: ', np.average(len_list))


sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=3000, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 2048
out_list, result_list = [], []
find_num = 0
f_w = open('./wikipedia/wikipedia_en_20240320_filter.json', 'w')

for index in tqdm(range(0, len(prompts_list), batch_size)):
    out_list_i = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)

    for info, events in zip(info_list[index: index+batch_size], out_list_i):
        info['event_list'] = events
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

f_w.close()

