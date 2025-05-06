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
from sentence_transformers.util import cos_sim
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import sys



def extract_answer(text):
    pattern = re.compile(r'Yes', re.IGNORECASE)
    if bool(pattern.search(text)):
        return True
    else:
        return False

def generate_vllm(llm, prompts, sampling_params):
    result_list = []
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        answer = extract_answer(generated_text)
        result_list.append(answer)

    return result_list

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='xx/llama3-8b', help="do predict")
parser.add_argument("--dataset", type=str, default='xx/result.json', help="do predict")
parser.add_argument("--save_dir", type=str, default='xx/result_lrae.json')
args = parser.parse_args()

prompt = "Question: {question} \nLable: {label} \n Prediction: {prediction} \nPlease judge whether the prediction is correct. If the prediction match label clearly, then the prediction is correct, otherwise wrong. The answer (Yes or No) is: "


model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)
question_list, prompts_list = [], []
len_list = []
for index, info in select_lines.iterrows():
    title = info['title']
    label = info['label']
    background = info['known_timeline']
    question = info['question']
    prediction = info['text']
    background_str = ''
    for i, known_event in enumerate(background):
        background_str += '%i. %s\n'%(i+1, known_event.strip())
    prompt_i = copy.deepcopy(prompt).format(question=question, prediction=prediction, label=label)
    messages = [{"role": "user", "content": prompt_i}]
    prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    question_list.append(info)
    prompts_list.append(prompt_chat_i)


if 'llama' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
elif 'Yi' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1)
llm = LLM(model=model_path, tensor_parallel_size=2, quantization=None, dtype='float16', trust_remote_code=True)

batch_size = 4096
out_list, result_list = [], []
acc_num = 0
all_num = 0
f_w = open(args.save_dir, 'w', encoding='utf-8')
for index in tqdm(range(0, len(prompts_list), batch_size)):
    out_list_i = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)

    for question, out in zip(question_list[index: index+batch_size], out_list_i):
        question['result'] = out
        all_num += 1
        if out == True:
            acc_num += 1
        f_w.writelines(json.dumps(question, ensure_ascii=False) + '\n')

f_w.close()

metric = {'acc': acc_num / all_num}
print(args.test_model)
print(metric)

