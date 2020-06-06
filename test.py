from SifRankZh import SIFRank
from ckiptagger import WS, POS
from elmoformanylangs import Embedder
import time
import json


with open('./data/data_seg_ptt.json') as f:
    data = json.load(f)

with open('./data/data_seg_dcard.json') as f:
    data += json.load(f)


ws = WS("./model/ckip", disable_cuda=False)
pos = POS('./model/ckip', disable_cuda=False)
ELMO = Embedder('model/elmo_tw')
sifrank = SIFRank(ELMO, ws, pos, word_freq_file='data/freq.txt')

batch = 8
idx = 100

result = []
start = time.time()
data = [d[:500] for d in data]
while idx < 110:
    kp = sifrank.extract_keyphrases(data[idx:idx+batch])
    tmp = list(
        map(lambda x, y: {'keyphrase': x, 'text': ''.join(y)}, kp, data[idx:idx+batch]))
    result.extend(tmp)
    idx += batch

with open('result.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False)


print(time.time() - start)