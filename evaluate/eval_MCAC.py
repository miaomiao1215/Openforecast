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
from transformers import AutoModelForMultipleChoice

def extract_answer(text):
    candidate_labels = ['A', 'B', 'C', 'D']
    pattern = pattern = r'\(([^)]*)\)'
    matches = re.findall(pattern, text.upper().replace('（', '(').replace('）', ')'))
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


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./test_results', help="do predict")
parser.add_argument("--model", type=str, default='xx/llama3_8b', help="do predict")
parser.add_argument("--test_category", type=str, default='MCAC', help="do predict")
parser.add_argument("--dataset", type=str, default='../dataset/test.json', help="do predict")

args = parser.parse_args()
prompt = "Based on the given event background and question, please select the most likely option from the four candidate options. Your final answer should be a single option letter, in the form (option letter) such as (A), at the end of your response. \n Event background: \n{background} \n Question: {question}\nEvent options: \n{options}\n Answer: "

lines = open(args.dataset, 'r', encoding='utf-8').readlines()
args.save_dir = os.path.join(args.save_dir, '%s/%s.json'%(os.path.basename(args.model), args.test_category))
os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)

question_list, prompts_chat_list = [], []
candidate_labels = ['A', 'B', 'C', 'D']
len_list = []
for line in lines:
    info = json.loads(line)
    article = ''
    title = info['title'] if type(info['title']) == str else info['title'][0]
    for question in info['MCAC']:
        question['title'] = title
        question['event_list'] = info['event_list']
        question['time'] = info['time']
        question['type'] = info['type']
        question_args = question['question']
        options = question['choices']
        label = candidate_labels[question['label']]
        question['label'] = label
        background = question['known_timeline']
        background_str = ''
        for i, known_event in enumerate(background):
            background_str += '%i. %s\n'%(i+1, known_event.strip())
        options_str = ''
        for i, option in enumerate(options):
            options_str += '(%s). %s\n'%(candidate_labels[i], option.strip())
        
        question_list.append(question)
        prompt_i = copy.deepcopy(prompt).format(background=background_str, question=question_args, options=options_str)
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
f_w = open(args.save_dir, 'w', encoding='utf-8')
f_w_metric = open(args.save_dir.replace('.json', '_metric.json'), 'w', encoding='utf-8')
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
f_w.close()
acc_dict = {'all_num': all_num, 'true_num': acc_num, 'acc': acc_num/all_num}
f_w_metric.writelines(json.dumps(acc_dict, ensure_ascii=False))
f_w_metric.close()
