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
rompt_1 = "You are a event timeline assistant. Based on the following article, please extract the complete event timeline in the article and sequentially output the key objective events in chronological order. The event timeline should contain the background, development, aftermath, investigation, and reactions if they are introduced in the article. If there is no content related to the event timeline, please directly output \"None\". The timeline output format should be: 1. On February 21, 2022, Putin announced that the Russian government would diplomatically recognize the Donetsk and Luhansk people's republics; 2. On February 22, 2022, the Federation Council unanimously authorised Putin to use military force outside Russia; etc. \n The article title: {title}\n The article: {article}\n The complete event timeline in chronological order is: "

lines = open(args.dataset, 'r').readlines()

info_list, prompts_list, prompts_list_chat, cluster_id_list = [], [], [], []
select_sections = ['Introduction', 'Background', 'Results', 'Aftermath', 'Events', 'Overview', 'Summary', 'Reactions', \
                               'Investigation', 'Impact', 'Timeline', 'Event', 'Incident', 'Accident', 'Reaction', 'Development', 'Response', 'Effects', \
                                'Description', 'Causes', 'Cause', 'Storylines', 'Incidents', 'Consequences', 'Responses', 'Protests', 'Massacre', \
                                'Outcome', 'Attacks']
len_list, timeline_num = [], []
model_path = "xx/llama3-70b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

for line in lines:
    info = json.loads(line)
    if info ['title'] not in select_titles:
        continue
    article = ''
    info_list.append(info)
    timeline_num.append(len(info['event_list']))
    for section, text in zip(info['section_titles'], info['section_texts']):

        if section.strip() in ['References', 'See also', 'External links', 'Notes', 'Further reading', 'Electoral system', 'Bibliography']:
            continue
        article += '== %s ==\n %s \n'%(section.strip(), text.strip())
    
    if article.count(' ') > 6000:
        article = ''
        for section, text in zip(info['section_titles'], info['section_texts']):
            if section.strip() in select_sections:
                article += '== %s ==\n %s \n'%(section.strip(), text.strip())
    len_list.append(article.count(' '))
    prompt_i = copy.deepcopy(prompt_1).format(title=info['title'], article=article)
    prompts_list.append(prompt_i)
    messages = [{"role": "user", "content": prompt_i}]
    prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompts_list_chat.append(prompt_chat_i)

print(' select num: ', len(info_list))
print(' average length: ', np.average(len_list))
print('original average timeline num: ', np.average(timeline_num))

if 'llama3' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 2048
out_list, result_list, timeline_num_new = [], [], []
find_num = 0
f_w = open('./wikipedia/wikipedia_en_20240320_filter.json', 'w', encoding='utf-8')

for index in tqdm(range(0, len(prompts_list), batch_size)):
    if 'llama3' in model_path:
        out_list_i = generate_vllm(llm, prompts_list_chat[index: index+batch_size], sampling_params)
    else:
        out_list_i = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)

    for info, events in zip(info_list[index: index+batch_size], out_list_i):
        info['event_list'] = events
        timeline_num_new.append(len(events))
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

f_w.close()
print('original average timeline num: ', np.average(timeline_num))
print('average timeline num: ', np.average(timeline_num_new))
