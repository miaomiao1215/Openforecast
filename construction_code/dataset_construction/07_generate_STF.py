import os
import json
import random
from tqdm import tqdm
import re
import random
from collections import defaultdict
from datetime import datetime
import pdb

lang = 'en'
lines = open('data.json', 'r').readlines()
f_w_train = open('STF_trainset.json', 'w')
f_w_dev = open('STF_devset.json', 'w')
f_w_test = open('STF_testset.json', 'w')
STF_train_num, STF_dev_num, STF_test_num = 0, 0, 0

split_time = datetime.strptime('2023-06-30', '%Y-%m-%d')
split_dev_time = datetime.strptime('2023-08-31', '%Y-%m-%d')
for line in tqdm(lines):
    info = json.loads(line)
    try:
        if len(info['time']) == 10:
            event_time = datetime.strptime(info['time'], '%Y-%m-%d')
        elif len(info['time']) == 7:
            event_time = datetime.strptime(info['time'], '%Y-%m')
        elif len(info['time']) == 4:
            event_time = datetime.strptime(info['time'], '%Y')
        else:
            event_time = datetime.strptime('2000-01-01', '%Y-%m-%d')
    except:
        event_time = datetime.strptime('2000-01-01', '%Y-%m-%d')

    all_event_list = info['event_list']
    time_list = info['atomic_time']
    time_events_dict = defaultdict(list)
    for time_i, event in zip(time_list, all_event_list):
        if time_i == None:
            continue
        if len(time_i) >= 10:
            try:
                time_date = datetime.strptime(time_i[0:10], '%Y-%m-%d')
            except:
                continue
            if event_time > split_time:
                if time_date > split_time:
                    time_events_dict[time_i].append(event)
            else:
                time_events_dict[time_i].append(event)

    if len(time_events_dict.keys()) == 0:
        info['STF'] = []
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
        continue
    test_times_candidate = []
    for time_i in time_events_dict.keys():
        if time_list.index(time_i) >= 2:
            test_times_candidate.append(time_i)
    test_time_list = random.sample(test_times_candidate, min(len(test_times_candidate), 2))
    # if len(test_time_list) == 0:
    #     pdb.set_trace()

    info['STF'] = []
    for test_time in test_time_list:
        target_events = time_events_dict[test_time]
        known_index_max = time_list.index(test_time)
        known_index_min = max(1, known_index_max-1)
        random_knwon_index = random.sample(range(known_index_min, known_index_max), 1)[0]+1

        known_timeline = all_event_list[0: known_index_max]
        info['STF'].append({'known_timeline': known_timeline, 'label': target_events, 'time': test_time})
        if event_time < split_time:
            STF_train_num += 1
            f_w_train.writelines(line)
        elif event_time < split_dev_time:
            STF_dev_num += 1
            f_w_dev.writelines(line)
        else:
            STF_test_num += 1
            f_w_test.writelines(line)

print('train_num: ', train_num)
print('dev_num: ', dev_num)
print('test_num: ', test_num)
print('STF_train_num: ', STF_train_num)
print('STF_dev_num: ', STF_dev_num)
print('STF_test_num: ', STF_test_num)
