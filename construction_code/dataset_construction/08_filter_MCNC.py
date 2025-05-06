import os
from tqdm import tqdm
import json
import random
import copy

def check_event(test_events, events):
    test_events_copy = copy.deepcopy(test_events)
    for test_event in test_events_copy:
        if 'Time' in test_event.keys():
            del test_event['Time']
        if test_event not in events:
            return True
    return False

lines = open('xx/MCNC_sturcture.json', 'r').readlines()
f_w = open('xx/MCNC_sturcture_filter.json', 'w')

bad_num = 0
filter_num = 0
for line in tqdm(lines):
    info = json.loads(line)
    event_list_extraction = info['event_list_extraction']

    multiqa_filter, multiqa_extraction_filter = [], []
    for question, question_extract in zip(info['MCNC'], info['MCNC_structure']):
        known_timeline = question['known_timeline']
        event_list_extraction_flat = []
        for events in event_list_extraction[len(known_timeline):]:
            event_list_extraction_flat.extend(events)
        for i, event in enumerate(event_list_extraction_flat):
            if 'Time' in event.keys():
                del event_list_extraction_flat[i]['Time']
        noise_events = question_extract['noise_events']
        if len(noise_events) != 3:
            bad_num += 1
            continue
            
        bool_bad = False
        for noise_event in noise_events:
            if not check_event(noise_event, event_list_extraction_flat):
                bool_bad = True
                break
        if bool_bad:
            bad_num += 1
            continue
        filter_num += 1
        indexs = [0,1,2,3]
        random.shuffle(indexs)
        gold_index = indexs.index(0)
        choices = [question['true_event']] + question['noise_events']
        choices_shuffle = [choices[i] for i in indexs]
        choices_extraction = [question_extract['true_event']] + question_extract['noise_events']
        choices_extraction_shuffle = [choices_extraction[i] for i in indexs]
        question['choices'] = choices_shuffle
        question['label'] = gold_index
        question_extract['choices'] = choices_extraction_shuffle
        question_extract['label'] = gold_index

        del question['noise_events']
        del question_extract['noise_events']
        multiqa_filter.append(question)
        multiqa_extraction_filter.append(question_extract)
        
    info['MCNC'] = multiqa_filter
    info['MCNC_structure'] = multiqa_extraction_filter
    f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
f_w.close()
print('bad_num', bad_num)
print('filter_num', filter_num)
