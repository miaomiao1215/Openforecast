import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
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

def generate_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    out_list, result_list = [], []
    # pattern = re.compile(r'Yes', re.IGNORECASE)
    pattern = re.compile(r'Event', re.IGNORECASE)

    for output in outputs:
        generated_text = output.outputs[0].text
        out_list.append(generated_text)
        if bool(pattern.search(generated_text)):
            result_list.append(True)
        else:
            result_list.append(False)
    return out_list, result_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='./wikipedia/wikipedia_zh_20240320_filter.json', help="do predict")

args = parser.parse_args()
prompt = "Please categorize the given Wikipedia article. Candidate categories include: [People, Country, Region, Organization, Sports Competition, Entertainment, item Definition, Holiday, Event].\n Wikipedia article title: {title}\n The article: {article}\n Please determine the category of the given wikipedia article. The prediction category from [People, Country, Region, Organization, Sports Competition, Entertainment, item Definition, Holiday, Event] is:"

lines = open(args.dataset, 'r').readlines()

lines_select, prompts_list, cluster_id_list = [], [], []
len_list = []
select_sections = ['Introduction', 'Background', 'Description'] if 'en' in args.dataset else ['Introduction', '背景', '简介']
for line in lines:
    info = json.loads(line)
    article = ''
    # bool_filter = False
    # for section, text in zip(info['section_titles'], info['section_texts']):
    #     if section.strip() in ['Roster', 'Medalists', 'Previous season', 'Game summaries', 'Standings', 'Awards', 'Offseason', 'Personnel', 'Player statistics', 'Births', 'Teams', \
    #         'Winners', 'Awards and honors', 'Qualifying', 'Regular season', 'Competition format', 'Race results', 'Draft picks', 'Final', 'Medal table', 'Playoffs', \
    #             'Preseason', 'Competitions', 'Before Eurovision', 'At Eurovision', 'Race', 'Medal summary', 'Election results', 'Candidates']:
    #         bool_filter = True
    #         break
    # if bool_filter:
    #     continue
    lines_select.append(line)
    for section, text in zip(info['section_titles'], info['section_texts']):
        if section.strip() in select_sections:
            article += '== %s ==\n %s \n'%(section.strip(), text.strip())
    len_list.append(len(article))
    if len(article) > 5000:
        article = article[0: 5000] + '...'
    prompts_list.append(copy.deepcopy(prompt_1).format(title=info['title'], article=article))
print(' average length: ', np.average(len_list))
print('select number:', len(lines_select))
assert len(lines_select) == len(prompts_list)
lines = lines_select
sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=128, n=1)
llm = LLM(model="./llama3-8b", tensor_parallel_size=4, quantization=None, dtype='float16')

batch_size = 1024
out_list, result_list = [], []
find_num = 0
f_w = open(args.dataset.replace('_filter.json', '_filter.json'), 'w')

for index in tqdm(range(0, len(prompts_list), batch_size)):
    out_list_i, result_list_i = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
    out_list.extend(out_list_i)
    result_list.extend(result_list_i)

    for line, result, out in zip(lines_select[index: index+batch_size], result_list_i, out_list_i):
        if result:
            f_w.writelines(line)
            find_num += 1

    print('find number: ', find_num)

f_w.close()


