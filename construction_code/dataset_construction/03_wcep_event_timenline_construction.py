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
import ray
import pdb


def extract_event(output):
    event_list = []
    for out_i in output.outputs:
        text_i = out_i.text
        event_list = []
        if '1.' not in text_i:
            continue
        text_i_temp = text_i[text_i.index('1.')+2: ]
        for j in range(2, 100):
            try:
                if '\n%i.'%j in text_i_temp:
                    index_end = text_i_temp.index('\n%i.'%j)
                    event_list.append(text_i_temp[: index_end].strip())
                    text_i_temp = text_i_temp[index_end+len('\n%i.'%j): ]
                elif '%i.'%j in text_i_temp:
                    index_end = text_i_temp.index('%i.'%j)
                    event_list.append(text_i_temp[: index_end].strip())
                    text_i_temp = text_i_temp[index_end+len('%i.'%j): ]
                else:
                    event_list.append(text_i_temp.strip().split('\n')[0])
                    break
            except:
                pdb.set_trace()
        event_list_filter = []
        for event_i in event_list:
            if len(event_i) > 0:
                event_list_filter.append(event_i)
        if len(event_list_filter) > 0:
            return event_list_filter
    return event_list
        

def generate_vllm(llm, prompts, cluster_id_list, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    event_list_dict = {}
    for output, cluster_id in zip(outputs, cluster_id_list):
        event_list = extract_event(output)
        prompt = output.prompt
        event_list_dict[cluster_id] = event_list
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return event_list_dict

def bool_merge(event_list1, event_list2):
    find_num = 0
    for event_i in event_list1:
        for event_j in event_list2:
            if event_i.strip() == event_j.strip():
                find_num += 1
                break
    if find_num >= (len(event_list1) * 0.8):
        return False
    else:
        return True

def merge_timelines_prompt(tokenizer, prompt_template, timeline_1, timeline_2, date_1, date_2):
    timeline_str_1 = ''
    for i, news_i in enumerate(timeline_1):
        timeline_str_1 += '%i.%s\n'%(i+1, news_i)
    timeline_str_2 = ''
    for i, news_i in enumerate(timeline_2):
        timeline_str_2 += '%i.%s\n'%(i+1, news_i)
    prompt = copy.deepcopy(prompt_template).format(event_chain_time1=date_1, event_chain1=timeline_str_1, event_chain_time2=date_2, event_chain2=timeline_str_2)
    messages = [{"role": "user", "content": prompt}]
    prompt_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_chat

prompt_1 = "You are a news event timeline assistant. Based on the following news, please briefly extract the objective events (excluding irrelevant events and subjective events) and output in chronological order. The event descriptions for each event should be brief, accurate and complete, especially for numbers, names (replace pronoun with their corresponding names) and dates. The events output format should be: 1. brief event; 2. brief event; etc. Don't output other content.  \n The News release time: {date}\n The News title: {title}\n The news article: {article}\n The events in chronological order is: "
prompt_2 = "You are an event timeline completion assistant. Please complete the given event timeline based on the following news. Rules: \n 1. New events should only be supplemented into the given event timeline, without altering the descriptions of events already specified within the timeline.\n 2.Only add objectively occurring key events that are not included in the given event timeline.\n 3. The newly added events should be limited to a small number of key events, excluding irrelevant events and subjective events.\n 4. The newly added events encompass events preceding, during, and subsequent to the given chain of events, which should be added to the front, middle and end of the timeline respectively, keeping the completed event timeline in chronological order.\n 5. Event descriptions should be brief, accurate and complete, especially for numbers, names (replace pronoun with their corresponding names) and dates.\n 6. If there are no new events, simply output the given event timeline.\n 7. The events output format should be: 1. brief event; 2. brief event; etc. \nThe given event timeline before {event_chain_time} is: {event_chain}\n Now, given the following news: \n The News release time: {date}\n The News title: {title}\n The news article: {article}\n The completed event timeline in chronological order is : "
prompt_3 = "You are an event chain merging assistant. Please merge the second event timeline into the first event chain according to the following rules: \n 1. Event timeline should be sorted in chronological order; \n 2. Merge the coreference events into one event, and the merged event timeline should contain all other events of the given two event timelines. \n3. Do not modify the descriptions of events in the original event timelines; \n 4. The events output format should be: 1. event; 2. event; etc. \nThe first event timeline before {event_chain_time1} is: {event_chain1}\n The second event timeline before {event_chain_time2} is: {event_chain2}\n The event timeline after merging in chronological order is : "


complex_event_all = json.loads(open('./wcep_data/wcep_news_filter.json', 'r').read())

model_path = "xx/llama3_70b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts_first_stage, cluster_id_list, prompts_first_stage_chat = [], [], []
for cluster_id, news_list in complex_event_all.items():
    first_news = news_list[0]
    cluster_id_list.append(cluster_id)
    if first_news['article'] == first_news['summary']:
        prompt_i = copy.deepcopy(prompt_1).format(date=first_news['date'], title=first_news['summary'], article='Unknown')
    else:
        prompt_i = copy.deepcopy(prompt_1).format(date=first_news['date'], title=first_news['title'], article=first_news['article'])
    prompts_first_stage.append(prompt_i)
    messages = [{"role": "user", "content": prompt_i}]
    prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts_first_stage_chat.append(prompt_chat_i)


llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16')
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=4096, n=3, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])

print('================================processing first news================================')
event_list_dict = generate_vllm(llm, prompts_first_stage_chat, cluster_id_list, sampling_params)

for cluster_id, event_list in event_list_dict.items():
    complex_event_all[cluster_id][0]['event_list'] = event_list
f_w = open('./wcep_data/wcep_news_temp_stage1.json', 'w')
f_w.writelines(json.dumps(complex_event_all, ensure_ascii=False, indent=4))
f_w.close()

# using prompt_2 to extract other news
news_index = 1
max_event_len = 10
for news_index in range(1, max_event_len):
    print('================================processing the %i news================================'%news_index)
    prompts, cluster_id_list = [], []
    for cluster_id, news_list in complex_event_all.items():
        if len(news_list) <= news_index:
            continue
        news = news_list[news_index]
        cluster_id_list.append(cluster_id)
        former_event_list = []
        for news_j in news_list[news_index-1: news_index]:
            former_event_list.extend(news_j['event_list'])
        former_news_string = ''
        event_chain_time = news_list[news_index-1]['date']
        for i, former_news_i in enumerate(former_event_list):
            former_news_string += '%i.%s\n'%(i+1, former_news_i)
        if news['article'] == news['summary']:
            prompt_i = copy.deepcopy(prompt_2).format(date=news['date'], title=news['summary'], article='Unknown', event_chain=former_news_string, event_chain_time=event_chain_time)
        else:
            prompt_i = copy.deepcopy(prompt_2).format(date=news['date'], title=news['title'], article=news['article'], event_chain=former_news_string, event_chain_time=event_chain_time)
        messages = [{"role": "user", "content": prompt_i}]
        prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_chat_i)
        # prompts.append(copy.deepcopy(prompt_1).format(date=first_news['date'], title=first_news['title'], article=first_news['article']))

    if len(prompts) == 0:
        break
    event_list_dict = generate_vllm(llm, prompts, cluster_id_list, sampling_params)
    
    merge_prompts, cluster_id_list_merge = [], []
    for cluster_id, event_list in event_list_dict.items():
        merge_flag = bool_merge(complex_event_all[cluster_id][news_index-1]['event_list'], event_list)
        if not merge_flag:
            complex_event_all[cluster_id][news_index]['event_list'] = event_list
        else:
            timeline1 = complex_event_all[cluster_id][news_index-1]['event_list']
            timeline2 = event_list
            date1 = complex_event_all[cluster_id][news_index-1]['date']
            date2 = complex_event_all[cluster_id][news_index]['date']
            merge_prompt_chat = merge_timelines_prompt(tokenizer, prompt_3, timeline1, timeline2, date1, date2)
            merge_prompts.append(merge_prompt_chat)
            cluster_id_list_merge.append(cluster_id)
            
    if len(merge_prompts)>0:
        event_list_dict_merge = generate_vllm(llm, merge_prompts, cluster_id_list_merge, sampling_params)
        for cluster_id, event_list in event_list_dict_merge.items():
            complex_event_all[cluster_id][news_index]['event_list'] = event_list

    f_w = open('./wcep_data/wcep_news_temp_stage%i.json'%(news_index+1), 'w')
    f_w.writelines(json.dumps(complex_event_all, ensure_ascii=False, indent=4))
    f_w.close()
    os.system('rm -rf %s'%('./wcep_data/wcep_news_temp_stage%i.json'%(news_index)))

f_w = open('./wcep_data/wcep_news_events_llama.json', 'w')
f_w.writelines(json.dumps(complex_event_all, ensure_ascii=False, indent=4))
f_w.close()
os.system('rm -rf %s'%('./wcep_data/wcep_news_temp_stage%i.json'%(max_event_len)))



