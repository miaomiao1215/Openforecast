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
        try:
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
        except:
            text_i = text_i.replace(':', '：').split('：')[-1]
            event_list = text_i.split('\n')
        event_list_filter = []
        for event_i in event_list:
            if len(event_i) > 0:
                event_list_filter.append(event_i)
        if len(event_list_filter) > 0:
            return event_list_filter
    return event_list


def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list, result_list = [], []
    for output in outputs:
        generated_text = output.outputs[0].text
        events = extract_event(output)
        out_list.append(generated_text.strip())
        result_list.append(events)
    return out_list, result_list


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./test_results', help="do predict")
parser.add_argument("--model", type=str, default='xx/llama3_8b', help="do predict")
parser.add_argument("--test_category", type=str, default='STF', help="do predict")
parser.add_argument("--dataset", type=str, default='../dataset/test.json', help="do predict")
args = parser.parse_args()

prompt = "Based on the given event background, please predict the most likely events to occur at the given time. Note that the events description should be brief, accurate, and complete, especially the name (replace the pronouns with the corresponding name). \n Example: \n Event background: \n1.Starting from March 2021, Ukrainian forces intensified their troop deployments on the frontline and began frequent clashes with Donbas militants.\n2.From March to April 2021, Russia began a large-scale military buildup near the border.\n3.From October 2021 to February 2022, Russia and Belarus carried out a second buildup.\n4.Throughout, the Russian government repeatedly denied it had plans to attack Ukraine.\n 5. On 21 February at 22:35, Putin announced that the Russian government would diplomatically recognize the Donetsk and Luhansk people's republics.\n6. The same evening, Putin directed that Russian troops deploy into Donbas.\n7. On 22 February, the Federation Council unanimously authorised Putin to use military force outside Russia. \n8. The following day, Ukraine's parliament proclaimed a 30-day nationwide state of emergency and ordered the mobilisation of all reservists.\n9. Russia began to evacuate its embassy in Kyiv.\nThe most likely events on February 24 is: 1.Russia will launch a full-scale invasion of Ukraine.\n2.Zelenskyy will declare martial law and a general mobilisation of all male Ukrainian citizens between 18 and 60, who will be banned from leaving the country.\n\n Now, based on the following event background, please predict the events that are most likely to occur on {time}.\n Event background: \n{background} \n The most likely events on {time} is: "

lines = open(args.dataset, 'r').readlines()
args.save_dir = os.path.join(args.save_dir, '%s/%s.json'%(os.path.basename(args.model), args.test_category))
os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)
question_list, prompts_chat_list = [], []

len_list = []
for line in lines:
    info = json.loads(line)
    title = info['title'] if type(info['title']) == str else info['title'][0]
    all_event_list = info['event_list']
    time_list = info['time']
    assert len(all_event_list) == len(time_list)
    article = ''
    for question in info['STF']:
        question['title'] = title
        question['event_list'] = info['event_list']
        question['time'] = question['time']
        question['time_list'] = time_list
        question['type'] = info['type']
        question['title'] = title
        target_time = question['time']
        background = question['known_timeline']
        background_str = ''
        for i, known_event in enumerate(background):
            background_str += '%i. %s\n'%(i+1, known_event.strip())

        question_list.append(question)
        prompt_i = copy.deepcopy(prompt).format(background=background_str, time=target_time)

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

batch_size = 1024
out_list, result_list = [], []
acc_num = 0
all_num = 0
f_w = open(args.save_dir, 'w')
for index in tqdm(range(0, len(prompts_chat_list), batch_size)):
    out_list_i, result_list_i = generate_vllm(llm, prompts_chat_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)
    
    for question, out, events in zip(question_list[index: index+batch_size], out_list_i, result_list_i):
        question['pred'] = out
        question['pred_events'] = events
        f_w.writelines(json.dumps(question, ensure_ascii=False) + '\n')

f_w.close()

