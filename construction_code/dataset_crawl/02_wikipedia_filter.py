import json
from tqdm import tqdm
import re

def contain_year(category_list):
    pattern = r'\b(19[5-9]\d|20[0-9]\d)\b'
    for category in category_list:
        if re.search(pattern, category):
            return True
    return False

def contain_event(category_list):
    pattern = re.compile(r'event|incident', re.IGNORECASE)
    for category in category_list:
        if bool(pattern.search(category)):
            return True
    return False

def section_filter(section_list):
    section_filter_list = ['History', 'Background',' Results','Schedule','Overview','Development','Events','Plot summary','Summary','Synopsis','Records','Election results',\
    'Timeline','Schedule and results','Description and history','List','Treatment','Impact','Battle','Reactions','Causes','Major results','Influence','Investigation',\
        'Calendar','Accidents and incidents','Influences','Incidents','Storylines','Result','Evolution','Cause','Effects','Reaction','Event','Process','Incident','Accident']
    for section in section_list:
        if section.strip() in section_filter_list:
            return True
    return False

lines = open('wikipedia_en_20240320.json', 'r').readlines()
print(len(lines))
filter_num = 0
f_w = open('./wikipedia_en_20240320_filter.json', 'w')
for line in tqdm(lines):
    info = json.loads(line)
    category_list = info['category']
    section_list = info['section_titles']

    if contain_year(category_list) or contain_event(category_list):
        if section_filter(section_list):
            filter_num += 1
            f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
f_w.close()
print('filter events number: %i'%filter_num)
