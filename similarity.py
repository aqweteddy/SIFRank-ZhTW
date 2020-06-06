import numpy as np
import json
from tqdm import tqdm


def get_cos_dist(sent_embed, kp_embed, layer_weight=[1, 1, 1]):
    sent_embed = np.array(sent_embed)
    kp_embed = np.array(kp_embed)
    score = 0
    for i in range(3):
        a, b = sent_embed[i], kp_embed[i]
        cos_sim = a@b.T / (np.linalg.norm(a) * np.linalg.norm(b))
        score += cos_sim * layer_weight[i]
    score = score / 3
    return score

IDX = 300

with open('./data/result_embed_ptt.json') as f:
    data = json.load(f)

src = data[IDX]
results = []
for d in data:
    if d == src:
        continue
    sc = 0
    for kp in src['keyprhases']:
        sc += kp['score'] * get_cos_dist(kp['embed'], d['text_embed'])
    sc /= len(src['keyprhases'])
    results.append((d, sc))

results.sort(key=lambda x: x[1])
results = results[-10:]
print(src['text'])
for kp in src['keyprhases']:
    print(kp['kp'], end=' ')
print('')
for r in results:
    print(r[0]['text'])
    for kp in r[0]['keyprhases']:
        print(kp['kp'], end=' ')
    print()
    print(r[1])