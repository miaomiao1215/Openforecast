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


def extract_answer(text):
    candidate_labels = ['A', 'B', 'C', 'D']
    pattern = pattern = r'\(([^)]*)\)'
    matches = re.findall(pattern, text.upper())
    if len(matches) > 0:
        for match in matches[::-1]:
            if match.strip() in candidate_labels:
                return match.strip()
        return None
    return None

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []
    for output in outputs:
        generated_text = output.outputs[0].text
        answer = extract_answer(generated_text)
        out_list.append(answer)
    return out_list

def events_2_str(events, lang):
    event_args = ['Time', 'Location', 'Subject', 'Event', 'Object']
    event_str_list = []
    for event in events:
        event_str = ''
        for event_arg in event_args:
            if event_arg in event.keys():
                if event[event_arg] != None:
                    event_str += '%s: %s; '%(event_arg, event[event_arg])
        for key in event.keys():
            if not key in event_args:
                if event[key] != None:
                    event_str += '%s: %s; '%(key, event[key])
        event_str = '[%s]'%event_str[:-2]
        event_str_list.append(event_str)
    return ';'.join(event_str_list)
                

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./test_results', help="do predict")
parser.add_argument("--model", type=str, default='xx/llama3_8b', help="do predict")
parser.add_argument("--test_category", type=str, default='MCNC_structure', help="do predict")
parser.add_argument("--dataset", type=str, default='../dataset/test.json', help="do predict")

args = parser.parse_args()

prompt = "Based on the given structured event background, please select one option from the four candidate event options that is most likely to occur in the future. Your final answer should be a single option letter, in the form (option letter) such as (A), at the end of your response. \n Event background: \n{background} \n Event options: \n{options}\n Answer: "


model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)

test_category_list = ['MCNC_extract_new']

lines = open(args.dataset, 'r', encoding='utf-8').readlines()
save_dir = os.path.join(args.save_dir, '%s/%s.json'%(os.path.basename(args.model), test_category))
os.makedirs(os.path.dirname(save_dir), exist_ok=True)

question_list, prompts_chat_list = [], []
candidate_labels = ['A', 'B', 'C', 'D']
len_list = []
for line in lines:
    info = json.loads(line)
    article = ''
    for question in info['MCNC_extract']:
        options = question['choices']
        label = candidate_labels[question['label']]
        question['label'] = label
        background = question['known_timeline']
        background_str = ''
        for i, known_event in enumerate(background):
            background_str += '%i. %s\n'%(i+1, events_2_str(known_event, args.lang))
        options_str = ''
        for i, option in enumerate(options):
            options_str += '(%s). %s\n'%(candidate_labels[i], events_2_str(option, args.lang))
        
        question_list.append(question)
        prompt_i = copy.deepcopy(prompt).format(background=background_str, options=options_str)
        if ('llama' in model_path or 'qwen' in model_path or 'Yi' in model_path):
            messages = [{"role": "user", "content": prompt_i}]
            prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts_chat_list.append(prompt_chat_i)
        else:
            prompts_chat_list.append(prompt_i)
            
print('select number:', len(question_list))
if 'llama' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
elif 'Yi' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1)
if 'falcon' in model_path:
    llm = LLM(model=model_path, tensor_parallel_size=2, quantization=None, dtype='bfloat16')
else:
    llm = LLM(model=model_path, tensor_parallel_size=2, quantization=None, dtype='bfloat16', trust_remote_code=True)

batch_size = 2048
out_list, result_list = [], []
acc_num = 0
all_num = 0
f_w = open(save_dir, 'w', encoding='utf-8')
f_w_metric = open(save_dir.replace('.json', '_metric.json'), 'w', encoding='utf-8')
for index in tqdm(range(0, len(prompts_chat_list), batch_size)):
    out_list_i = generate_vllm(llm, prompts_chat_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)
    
    label_list = [question['label'] for question in question_list[index: index+batch_size]]
    for question, out in zip(question_list[index: index+batch_size], out_list_i):
        if question['label'] == out:
            acc_num += 1
        all_num += 1
        question['pred'] = out
        f_w.writelines(json.dumps(question, ensure_ascii=False) + '\n')

    print('Acc: %.4f'%(acc_num/all_num))
acc_dict = {'all_num': all_num, 'true_num': acc_num, 'acc': acc_num/all_num}
f_w_metric.writelines(json.dumps(acc_dict, ensure_ascii=False))
f_w.close()
f_w_metric.close()

