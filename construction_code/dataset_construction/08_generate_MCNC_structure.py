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


def extract_targets(text):
    pattern = r'\{([^}]*)\}'
    matches = re.findall(pattern, text)
    return matches

def contains_specified_format(text):
    if text in ['None', 'none']:
        return True
    elif 'unknown' in text or 'not specified' in text or 'no specific' in text:
        return True
    else:
        pattern = re.compile(r'no\s+\w+\s+specified', re.IGNORECASE)
        match = pattern.search(text)
        pattern2 = re.compile(r'no\s+location|no\s+time|no\s+subject|no\s+object', re.IGNORECASE)
        match2 = pattern2.search(text)
        return bool(match) or bool(match2)

def extract_event(events_str):
    split_lines = events_str.split('\n')
    event_line = ''
    for line in split_lines:
        if '{Event:' in line:
            event_line = line
            break
    event_list = []
    event_strs = extract_targets(event_line)
    for event_str in event_strs:
        try:
            event_str_temp = event_str.replace('；', ';').replace('：', ':')
            event = {}
            event_args = event_str_temp.split(';')
            for event_arg in event_args:
                if event_arg.strip() == '':
                    continue
                if not ':' in event_arg:
                    continue
                if contains_specified_format(event_arg.split(':')[1].strip()):
                    event[event_arg.split(':')[0].strip()] = None
                else:
                    event[event_arg.split(':')[0].strip()] = event_arg.split(':')[1].strip()
            event_list.append(event)
        except:
            continue

    if len(event_list) == 0 and len(event_line.strip()) == 0:
        event_list = [{'Event': events_str}]
    elif len(event_list) == 0:
        event_list = [{'Event': event_line}]
        
    return event_list

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []
    # pattern = re.compile(r'Yes', re.IGNORECASE)
    pattern = re.compile(r'Event', re.IGNORECASE)

    for output in outputs:
        generated_text = output.outputs[0].text
        events = extract_event(generated_text)
        out_list.append(events)
    return out_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='train.json', help="do predict")
parser.add_argument("--save_dir", type=str, default='xx/MCNC_structure.json', help="do predict")
args = parser.parse_args()
prompt_1 = "Please extract structured event information from the given event text. The rules are as follows: \n1. Structured event information includes event trigger words and parameters such as time, location, subject, and object. If the above parameters are not enough to express the event clearly, you can also use other elements such as announcement content, condemnation content, boycott content and other elements (excluding causes and consequences); \n 2. Event elements such as subject and object should use full names instead of Pronouns (such as he, she, they, etc.);\n3. Event text can contain multiple structured events. Please output events in chronological order. If there are no structured events, output \"None\". The format is as shown in the example. \n An example is as follows: \nEvent text: \n In 2014, Russian troops occupied Crimea, and Russia quickly annexed it. Putin stated that he would establish a Russian military task force in Crimea\nThe extracted structured event list is: [{Event: occupation; Time: 2014; Location: Crimea; Subject: Russian troops; Object: Crimea}, {event: annexation; time: 2014; subject: Russia; object: Crimea}, {event: make statement; time: 2014; subject: Putin; statement content: he would establish a Russian military task force in Crimea}]\n\n Now, please perform event extraction on the following event text. Event text:\n%s\nThe extracted structured event list is:\n"

lines = open(args.dataset, 'r').readlines()
model_path = "xx/llama3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

info_list, prompts_list, cluster_id_list = [], [], []
len_list = []
for line in tqdm(lines):
    info = json.loads(line)
    article = ''
    info_list.append(info)

    prompt_temp = []
    # for i, former_event_i in enumerate(info['event_list_new']):
    for i, question in enumerate(info['MCNC']):
        for noise in question['noise_events']:
            prompt_i = copy.deepcopy(prompt_1)%noise
            if 'llama3' in model_path:
                messages = [{"role": "user", "content": prompt_i}]
                prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_temp.append({'text': noise, 'prompt': prompt_chat_i})
            else:
                prompt_temp.append({'text': noise, 'prompt': prompt_i})

    prompts_list.append(prompt_temp)

if 'llama3' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 1024
find_num = 0
f_w = open(args.save_dir, 'w', encoding='utf-8')
for index in tqdm(range(0, len(prompts_list), batch_size)):
    prompts_list_sentecnces = []
    temp_j = 0
    noise_event_list = []
    for i, prompts in enumerate(prompts_list[index: index+batch_size]):
        for info in prompts:
            prompts_list_sentecnces.append(info['prompt'])
            noise_event_list.append(info['text'])

    out_list = generate_vllm(llm, prompts_list_sentecnces, sampling_params)
    out_list_processed = {}
    for noise_event, out_i in zip(noise_event_list, out_list):
        out_list_processed[noise_event] = out_i

    for i, info in enumerate(info_list[index: index+batch_size]):
        multiqa_extraction = []
        for i, question in enumerate(info['MCNC']):
            question_extract = copy.deepcopy(question)
            question_extract['known_timeline'] = info['event_list_extraction'][0: len(question['known_timeline'])]
            true_event_index = info['event_list'].index(question['true_event'])
            question_extract['true_event'] = info['event_list_extraction'][true_event_index]
            question_extract['noise_events'] = []
            for noise_event in question['noise_events']:
                question_extract['noise_events'].append(out_list_processed[noise_event])
            multiqa_extraction.append(question_extract)
        info['MCNC_structure'] = multiqa_extraction
            
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

f_w.close()

