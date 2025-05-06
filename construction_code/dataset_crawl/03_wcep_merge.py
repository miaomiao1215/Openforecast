import os
import json
from collections import defaultdict
import datetime
from langdetect import detect
from tqdm import tqdm

story_news_dict = defaultdict(list)
wiki_items_list = []

for year in tqdm(range(2002, 2025)):
    lines = open('./data/%i_news_new.json'%year, 'r').readlines()
    info_all_list = []
    for line in tqdm(lines):
        try:
            info = json.loads(line)
        except:
            continue
        if info in info_all_list:
            continue
        else:
            info_all_list.append(info)
        if len(info['news_list'])==0:
            title = info['text']
            article = info['text']
            url = None
        else:
            bool_find = False
            for news in info['news_list']:
                title = news['title']
                article = news['article']
                url = news['url']
                if article == None:
                    continue
                try:
                    lang_text = detect(article)
                except:
                    continue
                if article.count(' ') > 100 and lang_text == 'en':
                    bool_find = True
                    break
            if bool_find == False:
                title = info['text']
                article = info['text']
                url = None
                
        category = info['category']
        date = info['date']
        summary = info['text']
        stories = [story.replace('%E2%80%93', '-').replace('_', ' ') for story in info['stories']]
        for story in stories:
            if story not in wiki_items_list:
                wiki_items_list.append(story)
        if len(stories) == 0:
            stories = [summary]

        wiki_links = info['wiki_links']
        wiki_links_process = []
        # for wiki_link in wiki_links:
        #     wiki_links_process.append(wiki_link.replace('/wiki/', ''))
        #     if wiki_link.replace('/wiki/', '') not in wiki_items_list:
        #         wiki_items_list.append(wiki_link.replace('/wiki/', ''))
        for index in range(len(stories), 0, -1):
            story_news_dict['|||'.join(stories[0: index])].append({'date': date, 'summary': summary, 'title': title, 'article': article, 'url': url, \
                'category': category, 'wiki_links': wiki_links_process})


f_w = open('wiki_items_wcep.txt', 'w')
for wiki_item in wiki_items_list:
    f_w.writelines(wiki_item + '\n')
f_w.close()
print('wiki_item num: ', len(wiki_items_list))

f_w = open('wcep_news.json', 'w')
story_news_dict_filter = {}
news_num = 0
for story, news_list in story_news_dict.items():
    if len(news_list) == 1:
        if news_list[0]['summary'] == news_list[0]['article']:
            continue
        story_news_dict_filter[story] = news_list
        news_num += 1
    else:
        story_news_dict_filter[story] = news_list
        news_num += len(news_list)
print('ori story num: %i, filter story num: %i'%(len(story_news_dict.keys()), len(story_news_dict_filter.keys())))
print('average news num: ', news_num / len(story_news_dict_filter.keys()))
f_w.writelines(json.dumps(story_news_dict_filter, ensure_ascii=False, indent=4))
f_w.close()

f_w = open('wcep_stories.txt', 'w')
for story in story_news_dict_filter.keys():
    f_w.writelines(story + '\n')
f_w.close()
