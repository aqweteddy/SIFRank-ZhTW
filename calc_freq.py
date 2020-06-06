import json
from tqdm import tqdm
with open('./data/data_seg_ptt.json') as f:
    data = json.load(f)

with open('./data/data_seg_dcard.json') as f:
    data += json.load(f)


word2frq = {}
for d in tqdm(data):
    for word in d:
        if ' ' in word:
            continue

        word = word.strip()
        if word in word2frq:
            word2frq[word] += 1
        else:
            word2frq[word] = 1

word_frq = list(word2frq.items())
word_frq = sorted(word_frq, key=lambda a: a[1], reverse=True)

with open('freq.txt', 'w') as f:
    for key, val in word_frq:
        if val < 10:
            break
        f.writelines([key, ' ', str(val), '\n'])
