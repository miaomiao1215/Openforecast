import os
import json
import random
from tqdm import tqdm
from datetime import datetime

lines = open('data.json', 'r').readlines()
f_w_train = open('LTF_trainset.json', 'w')
f_w_dev = open('LTF_devset.json', 'w')
f_w_test = open('LTF_testset.json', 'w')
train_num, dev_num, test_num = 0, 0, 0
LTF_train_num, LTF_dev_num, LTF_test_num = 0, 0, 0
split_time = datetime.strptime('2023-06-30', '%Y-%m-%d')
split_dev_time = datetime.strptime('2023-08-31', '%Y-%m-%d')
for line in tqdm(lines):
    info = json.loads(line)
    try:
        if len(info['time']) == 10:
            time = datetime.strptime(info['time'], '%Y-%m-%d')
        elif len(info['time']) == 7:
            time = datetime.strptime(info['time'], '%Y-%m')
        elif len(info['time']) == 4:
            time = datetime.strptime(info['time'], '%Y')
        else:
            time = datetime.strptime('2000-01-01', '%Y-%m-%d')
    except:
        time = datetime.strptime('2000-01-01', '%Y-%m-%d')
    
    all_event_list = info['event_list']
    if len(all_event_list) < 5:
        continue
    if time < split_time:
        train_num += 1
        f_w_train.writelines(line)
    elif time < split_dev_time:
        dev_num += 1
        f_w_dev.writelines(line)
    else:
        test_num += 1
        f_w_test.writelines(line)

    try:
        if time < split_time:
            indexs = random.sample(range(len(all_event_list)*2//3, len(all_event_list)-1), 2)
        else:
            indexs = random.sample(range(len(all_event_list)*2//3 - 1, len(all_event_list)-1), 2)
    except:
        indexs = random.sample(range(len(all_event_list)*2//3 - 1, len(all_event_list)-1), 1)
    info['LTF'] = []
    for index in indexs:
        known_timeline = all_event_list[0: index]
        label = all_event_list[index: ]
        info['LTF'].append({'known_timeline': known_timeline, 'label': label})
        if time < split_time:
            LTF_train_num += 1
        elif time < split_dev_time:
            LTF_dev_num += 1
        else:
            LTF_test_num += 1
    f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
f_w.close()

f_w_train.close()
f_w_dev.close()
f_w_test.close()
print('train_num: ', train_num)
print('dev_num: ', dev_num)
print('test_num: ', test_num)
print('LTF_train_num: ', LTF_train_num)
print('LTF_dev_num: ', LTF_dev_num)
print('LTF_test_num: ', LTF_test_num)