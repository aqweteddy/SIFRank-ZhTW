from pymongo import MongoClient
import re
from tqdm import tqdm
from ckiptagger import WS, POS

cli = MongoClient('mongodb://user:1234@linux.cs.ccu.edu.tw:27018')
cur = cli['forum']['dcard']
data = [d for d in tqdm(cur.find({}, {'_id': False, 'title': True, 'text': True}).limit(200000))]
data = [d['title'] + ' ' + d['text'] for d in data]


def load_stop_words(file='stopwords.txt'):
    stopwords = []
    with open(file) as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    return stopwords


def remove_special_char(text: str):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                  '', text, flags=re.MULTILINE)
    # remove sent from ...
    text = text.split('--\nSent ')[0]
    # keep only eng, zh, number
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    text = rule.sub(' ', text)
    # print(text)
    text = re.sub(" +", " ", text)

    return text

import json
data = [remove_special_char(d) for d in tqdm(data)]
with open('data_dcard.json', 'w') as f:
    json.dump(data, f)
