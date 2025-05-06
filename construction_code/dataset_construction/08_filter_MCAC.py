import os
from tqdm import tqdm
import json
import random
import copy


lines = open('xx/MCAC.json', 'r').readlines()
f_w = open('xx/MCAC_filter.json', 'w')

bad_num = 0
filter_num = 0
for line in tqdm(lines):
    try:
        info = json.loads(line)
    except:
        continue
    
    multiqa_args_filter = []
    for question in info['MCAC']:
        true_answer = question['true_answer']
        noise_events = question['noise_answers']
        if len(noise_events) != 3:
            bad_num += 1
            continue
            
        if true_answer in noise_events:
            bad_num += 1
            continue
        filter_num += 1
        indexs = [0,1,2,3]
        random.shuffle(indexs)
        gold_index = indexs.index(0)
        choices = [question['true_answer']] + question['noise_answers']
        choices_shuffle = [choices[i] for i in indexs]

        question['choices'] = choices_shuffle
        question['label'] = gold_index
        question['true_event'] = true_answer
        del question['noise_answers']
        del question['true_answer']

        multiqa_args_filter.append(question)

    info['MCAC'] = multiqa_args_filter
    f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
f_w.close()
print('bad_num', bad_num)
print('filter_num', filter_num)
