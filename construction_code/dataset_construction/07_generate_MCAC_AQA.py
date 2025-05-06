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
    noise_answers = []
    add_len = len('noise answer a:')

    try:

        text_temp = text.lower().replace('：', ':')
        if 'event argument prediction question:' in text_temp:
            index_question = text_temp.index('event argument prediction question:') + len('event argument prediction question:')
        else:
            index_question = 0
        index_true = text_temp.index('true answer:')
        index_a = text.lower().index('noise answer a:')
        index_b = text.lower().index('noise answer b:')
        index_c = text.lower().index('noise answer c:')
        question = text[index_question: index_true].strip()
        true_answer = text[index_true+len('true answer:'): index_a].strip()

        noise_answers.append(text[index_a+add_len: index_b].strip())
        noise_answers.append(text[index_b+add_len: index_c].strip())
        noise_answers.append(text[index_c+add_len: ].split('\n')[0].strip())
        noise_answers_clean = []
        for answer in noise_answers:
            if len(answer) > 0 and answer != true_answer:
                noise_answers_clean.append(answer)
        if len(noise_answers_clean) >= 2:
            return {'question': question, 'true_answer': true_answer, 'noise_answers': noise_answers_clean}
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
parser.add_argument("--dataset", type=str, default='train.json', help="do predict")
parser.add_argument("--save_dir", type=str, default='xx/train_MCAC.json', help="do predict")
args = parser.parse_args()
prompt_1 = "According to the given known event timeline, the target evolution event, and other subsequent evolution events, please design an event prediction question on the {args} argument of the given target evolution event, and generate the true answer and three candidate noise answers for this question. The rules are as follows:\n1.The generated question should focus on the {args} argument of the given target evolution event and include all other argument of the given target evolution event except for the {args} argument.\n2.Based on the generated question, extract the true answer from the target evolution event.\n3.The three candidate noise answers should be challenging and you can involve event argument from given event timeline or other subsequent evolution events, but the noise answers should be explicitly not occurred (not occurred in both given event timeline and other subsequent evolution events).\n4.The output format should be as follows: \nEvent argument prediction question: xxx\n True answer: xxx\n Noise answer A: xxx\n Noise answer B: xxx\n Noise answer C: xxx\n Example 1:\nKnown event timeline:\n1.Starting from March 2021, Ukrainian forces intensified their troop deployments on the frontline and began frequent clashes with Donbas militants.\n2.From March to April 2021, Russia began a large-scale military buildup near the border.\n3.From October 2021 to February 2022, Russia and Belarus carried out a second buildup.\n Target evolution event:\n In February 2022, Russia launched a full-scale invasion of Ukraine.\n Other subsequent evolution events:\n['In February 2022, Russia launched a full-scale invasion of Ukraine', 'On March 2nd, Russian forces captured the strategically important southern Ukrainian city of Kherson']\n Generated event subject argument prediction question and candidate answers:\n Event argument prediction question: In February 2022, who might initiate a full-scale invasion of Ukraine? \n True answer: Russia \n Noise answer A: Belarus. Noise answer B: Russia and Belarus. \n Noise answer C: Donbas militants.\n Example 2 (use same known event timeline, target evolution event, and other subsequent evolution events as example 1):\n Generated event type/behavior argument prediction question and candidate answers:\n Event argument prediction question: In February 2022, what might Russia do to Ukraine? \n True answer: Launch a full-scale invasion. \n Noise answer A: Conduct minor attacks. \n Noise answer B: Withdraw troops from the border. \n Noise answer C: Engage in diplomatic negotiations. \n Example 3 (use same known event timeline, target evolution event, and other subsequent evolution events as example 1): \n Generated event time argument prediction question and candidate answers: \n Event argument prediction question: When might Russia initiate a full-scale invasion of Ukraine? \n True answer: February 2022 \n Noise answer A: March 2022. \n Noise answer B: February 2023. \n Noise answer C: October 2021. \n Now, based on the above rules and examples, please generate a {args} argument prediction questions and candidate answers for the following event: \n Known event timeline: \n{timeline} \n Target evolution event: \n{true_event} \n Other subsequent evolution events: \n{evolution} \n Generated event {args} argument prediction question and candidate answers:"

model_path = "xx/llama3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

lines = open(args.dataset, 'r').readlines()
info_list,prompts_list,prompts_list_chat,len_list,sample_list = [],[],[],[],[]
args_list = ['Time', 'Location', 'Subject', 'Object']
question_num_dict = defaultdict(int)
question_num = 0
for line in tqdm(lines):
    info = json.loads(line)
    article = ''
    event_list = info['event_list']
    event_list_extraction = info['event_list_extraction']
    if len(event_list) < 5:
        continue
    args_list_candidate = ['type/behavior']
    args_text_dict = defaultdict(list)
    for events_dict_list in event_list_extraction:
        for event_dict in events_dict_list:
            for arg in args_list:
                if arg in event_dict:
                    if event_dict[arg] != None:
                        args_text_dict[arg].append(event_dict[arg])
    for arg in args_list:
        if len(list(args_text_dict[arg])) >= 3:
            args_list_candidate.append(arg)
    
    try:
        if 'train' in args.dataset:
            known_indexs = random.sample(range(len(event_list)*2//3, len(event_list)-1), 1)
        else:
            known_indexs = random.sample(range(len(event_list)*2//3, len(event_list)-1), 1)
    except:
        continue
    prompts_temp, prompts_chat_temp, sample_list_temp = [],[],[]
    for known_index in known_indexs:
        known_events = event_list[0:known_index]
        known_timeline = ''
        for i, former_event_i in enumerate(known_events):
            known_timeline += '%i.%s\n'%(i+1, former_event_i)
            
        evolution_events = event_list[known_index: ]
        true_event_indexs = random.sample(range(known_index, len(event_list)), min(2, len(event_list)-known_index))
        # true_event_indexs = [known_index]
        for true_event_index in true_event_indexs:
            true_event = event_list[true_event_index]
            
            args_list_candidate_2 = ['type/behavior']
            for event_dict in event_list_extraction[true_event_index]:
                for arg in event_dict.keys():
                    if not arg in args_list_candidate_2 and event_dict[arg] != None:
                        args_list_candidate_2.append(arg)
            args_list_candidate_filter = []
            for arg_candidate in args_list_candidate:
                if arg_candidate in args_list_candidate_2:
                    args_list_candidate_filter.append(arg_candidate)
            
            if random.random() > 0.5:
                if 'Time' in args_list_candidate_filter:
                    args_list_candidate_filter.remove('Time')
            question_arg = random.sample(args_list_candidate_filter, 1)[0]
            question_num_dict[question_arg] += 1
            prompt_i = copy.deepcopy(prompt_1).format(timeline=known_timeline, evolution=evolution_events, true_event=true_event, args=question_arg)
            messages = [{"role": "user", "content": prompt_i}]
            prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            len_list.append(prompt_chat_i.count(' '))
            prompts_temp.append(prompt_i)
            prompts_chat_temp.append(prompt_chat_i)
            sample_list_temp.append({'question_type': question_arg, 'known_timeline': known_events, 'true_event': true_event})
    info_list.append(info)
    prompts_list_chat.append(prompts_chat_temp)
    question_num += len(prompts_chat_temp)
    prompts_list.append(prompts_temp)
    sample_list.append(sample_list_temp)
print(' average length: ', np.average(len_list))
print(' sample number: ', question_num)
print('question_num_dict:', question_num_dict)

if 'llama3' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=4096, n=1)
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
    if ('llama' in model_path or 'qwen' in model_path or 'Yi' in model_path) and 'qwen1.5_32b' not in model_path:
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
        questions_list = out_list_processed[i]
        assert len(samples) == len(questions_list)
        samples_clean = []
        for sample, question_dict in zip(samples, questions_list):
            if question_dict != None:
                sample.update(question_dict)
                samples_clean.append(sample)
                sample_num += 1
            else:
                continue
        if len(samples_clean) > 0:
            info['MCAC'] = samples_clean
            f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')

f_w.close()
print('sample_num: ', sample_num)
