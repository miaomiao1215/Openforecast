import pandas as pd
import feapder
from newspaper import Article
import json
from tqdm import tqdm
import datetime
from urllib.parse import urlparse
import csv
from collections import defaultdict
from newspaper import Config
import requests
from urllib.parse import urlparse
from newspaper import fulltext
from newsplease import NewsPlease
import sys
import os
import time

def extract_main_website(url):
    parsed_url = urlparse(url)
    main_website = parsed_url.netloc
    # if '.' in main_website:
    #     main_website = main_website.split('.')[-2] + '.' + main_website.split('.')[-1]
    return main_website


def crawl_other(url, summary):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
       'Accept-Encoding': 'gzip, deflate, br',
       'Referer': 'https://www.reuters.com/redirect/',
       'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
       'Connection': 'keep-alive',
       'Upgrade-Insecure-Requests': '1',
       'Sec-Fetch-Dest': 'document',
       'Sec-Fetch-Mode': 'navigate',
       'Sec-Fetch-Site': 'same-origin',
       'Sec-Fetch-User': '?1',
       'If-Modified-Since': 'Mon, 29 Apr 2024 06:31:25 GMT',
       'If-None-Match': 'W/"d6304-/OF4w1Yv3pykLRl8WNZWcUriM/I"',
       'TE': 'trailers',
       'Cookie':'_ga_WBSR7WLTGD=GS1.1.1714372789.1.1.1714372844.5.0.0; _ga=GA1.1.468103864.1714372790; _fssid=272f6aca-9dfd-4b55-a3f8-93e74232b7ee; datadome=IlmfcfjHxEOOULPf7pLyDlf72fSbyJqHPsI~iV5XNSsPh06RjKX9z_nBxtaILnYMbZnLCApd6cpR7ZzNpyqLMgrssHI1qYg3ZvlNyhdEkh4aGsZSfXFwkogbbsa83PUc; reuters-geo={"country":"HK", "region":"-"}; RT="z=1&dm=reuters.com&si=ejuxpygg4yd&ss=lvklahpa&sl=0&tt=0"; cleared-onetrust-cookies=Thu, 17 Feb 2022 19:17:07 GMT; OptanonConsent=isGpcEnabled=0&datestamp=Mon+Apr+29+2024+14%3A40%3A51+GMT%2B0800+(%E4%B8%AD%E5%9B%BD%E6%A0%87%E5%87%86%E6%97%B6%E9%97%B4)&version=202310.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=f36cba26-fd88-420f-bb0f-503626436cfa&interactionCount=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2CSPD_BG%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false&geolocation=HK%3B; usprivacy=1---; _gcl_au=1.1.1022458304.1714372785; _dd_s=rum=0&expire=1714373749494; _awl=2.1714372851.5-18025a8b78babaf46b72fe10d7787852-6763652d617369612d6561737431-0; _cb=OpKXNBn1rYuCqU5o_; _chartbeat2=.1714372786535.1714372833768.1.DbeDYACDYjk_XY5KeBgMhhSDlM_a1.4; _cb_svref=external; _lr_geo_location_state=; _lr_geo_location=HK; _fbp=fb.1.1714372788061.413485064; cookie=cc17e485-4665-4f43-82cf-84d1ef88c2e7; cookie_cst=zix7LPQsHA%3D%3D; _lr_retry_request=true; _lr_env_src_ats=false; ajs_anonymous_id=2abed210-7814-4422-9af9-2a44e60b4588; pbjs_fabrickId=%7B%22fabrickId%22%3A%22E1%3A-rIuTgVPwh2hYPzWM3D0e4q50NDId5e-Th2ZGT-U5BI1QtJxVFIwnhm_JABtrvLgRkwMGKuXR9EYuZYuiyrElkA55CFIszleLMdThYUGOYG-q5BS5JU6FzVqw_CkT8PQ%22%7D; pbjs_fabrickId_cst=zix7LPQsHA%3D%3D; cto_bundle=3yM8Dl9FVXJNZ1pzUzZDcTdramx1bjBaN1dBcWJENnZOQlFWcGMwMXglMkJ1WWZKRHhXZ21Ub1FwRmFscEV2akdqTENjNDlhWHpFUiUyRjdQQTZnRTRIOTZTRWpYR3FCaTZKcXozRWxoMW9Mc2tQaU9UY25UQm9MJTJGNCUyQmI0RnFRUnFkZmhUclI2V2cxRyUyQmE0UGcwMXolMkYxRkJidzA5Q0ElM0QlM0Q; cto_bidid=pMwZOl9NcjJweVQwclI5akdlNSUyRlN5TjQlMkJrcVM1ZXVMRG85Sjc5dUNUJTJGOEVTV3MzTWRWNXE3Y2t5aEkwc2d2WGRGdTB1RFdpMlVwemV1MURvMjMzMyUyRmpoc1VCOHEwSUNXbDZSciUyQktIRGZyNWNFR3MlM0Q; _au_1d=AU1D-0100-001714372790-RTQW075E-FFHP; OptanonAlertBoxClosed=2024-04-29T06:40:51.209Z; ABTastySession=mrasn=&lp=https%253A%252F%252Fwww.reuters.com%252Faccount%252Fregister%252Fsign-up%252F%253Fredirect%253Dhttps%253A%252F%252Fwww.reuters.com%252Farticle%252Fus-iraq-security-usa%252Firaqi-militias-start-withdrawing-from-u-s-embassy-idUSKBN1Z01N9%252F%253Fil%253D0%2526referrer%253Dregistration_button%2526journeyStart%253Dnavigation; _ga=GA1.1.468103864.1714372790; _gid=GA1.2.1520863861.1714372797; uuid=f0afcc4b-71a4-497a-901d-f2b88532fd57; OneTrustWPCCPAGoogleOptOut=false; __qca=P0-1191635149-1714372828437; __gads=ID=a8b5af4488a06341:T=1714372832:RT=1714372832:S=ALNI_MYAJ7rLGnwk6GT8AtEzpHF2VXS4EQ; __gpi=UID=00000dff3496003a:T=1714372832:RT=1714372832:S=ALNI_MY2n-88MkcYkh9XkYTX2ngWflvalg; __eoi=ID=b27212d0ec0f260d:T=1714372832:RT=1714372832:S=AA-Afjbl-VJVERC6C0Ygn3gtHE_z; permutive-id=18f5b68f-159d-47b2-a819-4e0644e16742; _lr_sampling_rate=100; _ga_FVWZ0RM4DH=GS1.1.1714372846.1.0.1714372846.60.0.0; _chartbeat4=t=Cd-RVHBuq3CYCmuPqNB5jAIlJVbNG&E=6&x=0&c=0.21&y=8694&w=567; cto_dna_bundle=kjYfUl9qeW5XZHNpRTNLM0tybEM4YUlpViUyRmVZVmwwcFBWS2pqY1klMkJaTnV6ZGsxZXdrSk93R09yR29pTGNoYmJ6RlUyNGFJZjZ0VTdrNG9Da3hzY1Y1RTNLNnclM0QlM0Q'}
    html = requests.get(url, headers=headers, timeout=10).text
    content = fulltext(html)
    try:
        title = ' '.join(url.split('/')[-2].replace('-', ' ').split(' ')[0: -3])
    except:
        title = summary
    if 'Please enable JS and disable any ad blocker' in content:
        print('Please enable JS and disable any ad blocker')
        sys.exit(0)
    if content.count(' ') < 100:
        return None, None
    return title, content


def crawl_to_title_and_article(url, summary):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    article = Article(url)
    main_url = extract_main_website(url)
    if main_url in ['www.reuters.com', 'www.barrons.com', 'www.abc.net.au', 'english.alarabiya.net', 'www.arabnews.com', \
        'news.abs-cbn.com', 'www.aa.com.tr', 'abcnews.go.com', 'www.channelnewsasia.com']:
        title, content = crawl_other(url, summary)
        if len(content) < 10:
            article = NewsPlease.from_url(url, timeout=10)
            title = article.title
            content = article.maintext
    else:
        article.download()
        article.parse()
        title = article.title
        content = article.text

    return title, content

for year in range(2024, 2001, -1):
    print('======year: ', year)
    
    lines = open('./data/%i_news.json'%year, 'r').readlines()
    f_w = open('./data/%i_news_new.json'%year, 'w')
    find_num, all_num = 0, 0
    for line in tqdm(lines):
        info = json.loads(line)
        references = info['references']
        news_list = info['news_list']
        if len(news_list) > 0:
            f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
            continue
        info['news_list'] = []
        for url in references:
            all_num += 1
            time.sleep(1)
            try:
                title, article = crawl_other(url, info['text'])
                if article != None:
                    info['news_list'].append({'title': title, 'article': article, 'url': url})
                    find_num += 1
            except:
                try:
                    article = NewsPlease.from_url(url, timeout=6)
                    title = article.title
                    content = article.maintext
                    find_num += 1
                except:
                    continue
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
        # print('=========find %i/%i========='%(find_num, all_num))
    print('=========find %i/%i========='%(find_num, all_num))
    f_w.close()
