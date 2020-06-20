from ckiptagger import WS, POS
import json
from tqdm import tqdm


with open('data_dcard.json') as f:
    data = json.load(f)

pbar = tqdm(total=len(data))
start = 0
ws = WS('../model/ckip', disable_cuda=False)
output = []
batch = 128
while start < len(data):
    pbar.update(start)
    d = data[start:start+batch]
    start = start + batch
    output.extend(ws(d))
pbar.close()

with open('data_seg_dcard.json',  'w') as f:
    json.dump(output, f)
