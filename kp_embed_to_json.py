import json
from SifRankZh import SIFRank
from ckiptagger import WS, POS
from elmoformanylangs import Embedder
import time
from tqdm import tqdm
import logging


logging.getLogger('elmoformanylangs').setLevel(logging.WARNING)
with open('data/data_ptt.json') as f:
    data = json.load(f)
ws = WS("./model/ckip", disable_cuda=False)
pos = POS('./model/ckip', disable_cuda=False)
ELMO = Embedder('model/elmo_tw')
sifrank = SIFRank(ELMO, ws, pos, word_freq_file='freq.txt')

results = []
batch = 8
idx = 0

result = []
start = time.time()
data = [d[:500] for d in data]
pbar = tqdm(total=1000)
while idx < 1000:
    kp = sifrank.extract_keyphrases_with_embedding(data[idx:idx+batch])
    result.extend([dict(k) for k in kp])
    idx += batch
    pbar.update(batch)
print(time.time() - start)
with open('data/result_embed_ptt.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False)