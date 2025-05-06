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
    if 'answer' in text.lower():
        return text[text.lower().rfind('answer')+7: ].strip()
    else:
        return text

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []
    for output in outputs:
        generated_text = output.outputs[0].text
        out_list.append(extract_answer(generated_text))
    return out_list


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./test_results', help="do predict")
parser.add_argument("--model", type=str, default='xx/llama3_8b', help="do predict")
parser.add_argument("--test_category", type=str, default='AQA', help="do predict")
parser.add_argument("--dataset", type=str, default='../dataset/test.json', help="do predict")
args = parser.parse_args()

prompt = "Based on the given event background, please answer the given question with the most likely answer. Event background: \n{background} \n Question: {question}\n Answer: "

lines = open(args.dataset, 'r').readlines()

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
    for question in info['MCAC']:
        question['title'] = title
        question['event_list'] = info['event_list']
        question['time'] = info['time']
        question['type'] = info['type']
        question_args = question['question']
        label = question['choices'][question['label']]
        question['label'] = label
        background = question['known_timeline']
        background_str = ''
        for i, known_event in enumerate(background):
            background_str += '%i. %s\n'%(i+1, known_event.strip())

        question_list.append(question)
        prompt_i = copy.deepcopy(prompt).format(background=background_str, question=question_args)

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
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16', trust_remote_code=True)

batch_size = 2048
out_list, result_list = [], []
all_num = 0
f_w = open(args.save_dir, 'w')
for index in tqdm(range(0, len(prompts_chat_list), batch_size)):
    out_list_i = generate_vllm(llm, prompts_chat_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)
    
    label_list = [question['label'] for question in question_list[index: index+batch_size]]
    for question, out in zip(question_list[index: index+batch_size], out_list_i):
        question['pred'] = out
        f_w.writelines(json.dumps(question, ensure_ascii=False) + '\n')

f_w.close()

