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
import random

def split_noise(text, lang='en'):
    noise_events = []
    add_len = len('Noise Event A:')

    try:
        index_a = text.lower().index('noise event a:')
        index_b = text.lower().index('noise event b:')
        index_c = text.lower().index('noise event c:')

        noise_events.append(text[index_a+add_len: index_b].strip())
        noise_events.append(text[index_b+add_len: index_c].strip())
        noise_events.append(text[index_c+add_len: ].split('\n')[0].strip())
        noise_events_clean = []
        for event in noise_events:
            if len(noise_events) > 0:
                noise_events_clean.append(event)
        if len(noise_events_clean) >= 2:
            return noise_events_clean
        else:
            return None
    except:
        return None
    

def generate_vllm(llm, prompts, sampling_params, lang='en'):
    outputs = llm.generate(prompts, sampling_params)
    out_list = []

    for output in outputs:
        generated_text = output.outputs[0].text
        noise_events = split_noise(generated_text, lang)
        out_list.append(noise_events)
    return out_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='xx/train.json', help="do predict")
parser.add_argument("--save_dir", type=str, default='xx/train_MCNC.json', help="do predict")
args = parser.parse_args()
prompt_1 = "According to the given known event timeline, subsequent evolution events and one true candidate event, please generate three candidate noise events for the given true candidate event to the event forecasting problem. \nThe rules are as follows: \n1. The true candidate event comes from the given subsequent evolution events. The given subsequent evolution events is used to prevent the generation of candidate noise events that actually occur. \n2. The three candidate noise events should be challenging and similar to the given true candidate events. And the three candidate noise events should adopt the same time as the true candidate event, but they should be explicitly not occurred (according to the known events timeline and subsequent evolution events). \n3.You can generate candidate noise events that contradict the true event or have different event arguments, including variations in event type. \n4. The output format should be: Noise Event A: xxx\n Noise Event B: xxx\n Noise Event C: xxx\n\n An example is provided below: \n The known event timeline: \n1.From October 2021 to February 2022, Russia and Belarus carried out a second buildup.\n2.Throughout, the Russian government repeatedly denied it had plans to attack Ukraine.\n 3.On 21 February, Putin announced that the Russian government would diplomatically recognize the Donetsk and Luhansk people's republics\n4. Putin will direct that Russian troops deploy into Donbas.\n5. On 22 February, the Federation Council unanimously authorised Putin to use military force outside Russia. \n6. The following day, Ukraine's parliament proclaimed a 30-day nationwide state of emergency and ordered the mobilisation of all reservists. Russia began to evacuate its embassy in Kyiv.\n The subsequent evolution events: [\"In February 2022, Russia launched a full-scale invasion of Ukraine.\", \"Zelenskyy declared martial law and a general mobilisation of all male Ukrainian citizens between 18 and 60\"] \n The true candidate event: \nIn February 2022, Russia launched a full-scale invasion of Ukraine.\n The three candidate noise answers are: \nNoise Event A: In February 2022, Russia launched a small-scale attack on Ukraine. \nNoise Event B: In February 2022, Russia deploy troops into the Russian-Ukrainian border. \nNoise Event C: In February 2022, Russia and Belarus conduct joint military exercises to threaten Ukraine. \n\nNow, based on the above rules and example, please generate three candidate noise answers. The known event timeline: {timeline} \nThe subsequent evolution events: {evolution} \nThe true candidate event:{true_event}\nThe three candidate noise answers are: "

model_path = "xxx/llama3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

lines = open(args.dataset, 'r').readlines()
sample_num = 0
info_list,prompts_list,prompts_list_chat,len_list,sample_list = [],[],[],[],[]
for line in tqdm(lines):
    info = json.loads(line)
    article = ''
    event_list = info['event_list']
    try:
        if 'train' in args.dataset:
            known_indexs = random.sample(range(len(event_list)*2//3, len(event_list)-2), 1)
        else:
            known_indexs = random.sample(range(len(event_list)*2//3, len(event_list)-2), min(2, len(event_list)-1))
    except:
        continue
    prompts_temp, prompts_chat_temp, sample_list_temp = [],[],[]
    for known_index in known_indexs:
        known_events = event_list[0:known_index]
        known_timeline = ''
        for i, former_event_i in enumerate(known_events):
            known_timeline += '%i.%s\n'%(i+1, former_event_i)
            
        evolution_events = event_list[known_index: ]
        true_event_indexs = [known_index] + random.sample(range(known_index+1, len(event_list)), min(2, len(event_list)-known_index-1))
        # true_event_indexs = [known_index]
        for true_event_index in true_event_indexs:
            true_event = event_list[true_event_index]
            prompt_i = copy.deepcopy(prompt_1).format(timeline=known_timeline, evolution=evolution_events, true_event=true_event)
            messages = [{"role": "user", "content": prompt_i}]
            prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            len_list.append(prompt_chat_i.count(' '))
            prompts_temp.append(prompt_i)
            prompts_chat_temp.append(prompt_chat_i)
            sample_list_temp.append({'known_timeline': known_events, 'true_event': true_event})
            sample_num += 1
    info_list.append(info)
    prompts_list_chat.append(prompts_chat_temp)
    prompts_list.append(prompts_temp)
    sample_list.append(sample_list_temp)
print(' average length: ', np.average(len_list))
print(' sample number: ', sample_num)

if 'llama3' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1048, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1048, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 1024
out_list, result_list = [], []
find_num = 0
f_w = open(args.save_dir, 'w', encoding='utf-8')

sample_num = 0
for index in tqdm(range(0, len(prompts_list), batch_size)):
    prompts_list_sentecnces = []
    temp_j = 0
    index_dict = {}
    if 'llama3' in model_path:
        prompts_batch = prompts_list_chat[index: index+batch_size]
    else:
        prompts_batch = prompts_list[index: index+batch_size]
    for i, prompts in enumerate(prompts_batch):
        for prompt_j in prompts:
            prompts_list_sentecnces.append(prompt_j)
            index_dict[temp_j] = i
            temp_j += 1

    out_list = generate_vllm(llm, prompts_list_sentecnces, sampling_params, args.lang)

    out_list_processed = defaultdict(list)
    for i, out_i in enumerate(out_list):
        out_list_processed[index_dict[i]].append(out_i)

    for i, info, samples in zip(range(len(info_list[index: index+batch_size])), info_list[index: index+batch_size], sample_list[index: index+batch_size]):
        noise_events_list = out_list_processed[i]
        assert len(samples) == len(noise_events_list)
        for j, noise_events in enumerate(noise_events_list):
            if noise_events == None:
                samples[j]['noise_events'] = None
            else:
                noise_events_clean = []
                for noise_event in noise_events:
                    if noise_event.strip() != samples[j]['true_event'].strip():
                        noise_events_clean.append(noise_event)
                samples[j]['noise_events'] = noise_events_clean
        samples_clean = []
        for sample in samples:
            if sample['noise_events'] != None:
                if len(sample['noise_events']) >= 2:
                    samples_clean.append(sample)
                sample_num += 1
            else:
                continue
        info['MCNC'] = samples_clean
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

f_w.close()
print('sample_num: ', sample_num)
