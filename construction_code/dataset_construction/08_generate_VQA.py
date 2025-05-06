import os
import json
import copy
import random
from tqdm import tqdm

lines = open('./MCNC.json', 'r').readlines()
f_w = open('./VQA.json', 'w')
labels = [0,1,2,3]
all_num = 0
for line in tqdm(lines):
    info = json.loads(line)
    for i, question in enumerate(info['MCNC']):
        choices = question['choices']
        true_event = question['choices'][question['label']]
        candidate_copy = copy.deepcopy(labels)
        candidate_copy.remove(question['label'])
        false_index = random.sample(candidate_copy, 1)[0]
        false_event = question['choices'][false_index]
        if random.random() > 0.5:
            info['MCNC'][i]['VQA'] = {'event': true_event, 'label': True}
            all_num += 1
        else:
            info['MCNC'][i]['VQA'] = {'event': false_event, 'label': False}
            all_num += 1
    f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
print(all_num)
f_w.close()
