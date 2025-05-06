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
    out_list, result_list = [], []
    # pattern = re.compile(r'Yes', re.IGNORECASE)
    true_pattern = r"(Yes|will happen)"
    false_pattern = r"(\"No\"|is unlikely|will not happen)"
    for output in outputs:
        generated_text = output.outputs[0].text
        out_list.append(generated_text.strip())
        # if bool(pattern.search(generated_text)):
        if re.search(true_pattern, generated_text, re.IGNORECASE):
            result_list.append(True)
        elif re.search(false_pattern, generated_text, re.IGNORECASE):
            result_list.append(False)
        elif 'with high probability' in generated_text:
            result_list.append(True)
        else:
            result_list.append(False)
    return out_list, result_list


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./test_results', help="do predict")
parser.add_argument("--model", type=str, default='xx/llama3_8b', help="do predict")
parser.add_argument("--test_category", type=str, default='VQA', help="do predict")
parser.add_argument("--dataset", type=str, default='../dataset/test.json', help="do predict")

args = parser.parse_args()
prompt = "Event background: \n{background} \n Event option: \n{options}\n Based on the given event background, please predict whether the given event option will happen with high probability. Your final answer should be \"Yes\" or \"No\", at the end of your response.\n Answer (\"Yes\" or \"No\"): "

lines = open(args.dataset, 'r', encoding='utf-8').readlines()
args.save_dir = os.path.join(args.save_dir, '%s/%s.json'%(os.path.basename(args.model), args.test_category))
os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)
question_list, prompts_chat_list = [], []

len_list = []
for line in lines:
    info = json.loads(line)
    article = ''
    title = info['title'] if type(info['title']) == str else info['title'][0]
    for question in info['MCNC']:
        question['title'] = title
        question['event_list'] = info['event_list']
        question['time'] = info['time']
        question['type'] = info['type']
        background = question['known_timeline']
        label = question['VQA']['label']
        option = question['VQA']['event']
        background_str = ''
        for i, known_event in enumerate(background):
            background_str += '%i. %s\n'%(i+1, known_event.strip())

        question_list.append(question)
        prompt_i = copy.deepcopy(prompt).format(background=background_str, options=option)

        if ('llama' in model_path or 'qwen' in model_path or 'Yi' in model_path) and 'nofinetune' not in model_path:
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
    generate_text_list, out_list_i = generate_vllm(llm, prompts_chat_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)
    
    label_list = [question['VQA']['label'] for question in question_list[index: index+batch_size]]
    for question, out in zip(question_list[index: index+batch_size], out_list_i):
        if question['VQA']['label'] == out:
            acc_num += 1
        all_num += 1
        question['pred'] = out
        f_w.writelines(json.dumps(question, ensure_ascii=False) + '\n')

    print('Acc: %.4f'%(acc_num/all_num))
f_w.close()
acc_dict = {'all_num': all_num, 'true_num': acc_num, 'acc': acc_num/all_num}
f_w_metric.writelines(json.dumps(acc_dict, ensure_ascii=False))
f_w_metric.close()
