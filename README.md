# SIFRank For Zh-TW

##### `keyphrase extraction` `keyword extraction`

* 這是基於 [SIFRank](https://github.com/sunyilgdx/SIFRank_zh)的繁體中文修改版
* [論文](https://ieeexplore.ieee.org/document/8954611)

## 主要修改

* 將斷詞換成 ckip
* 移除 document segmentation，改為一次 predict 多篇文章
    * 主要是為了吃到 ckiptagger 可一次斷多篇文章，ELMO 也可吃到
* embeddings alignment 尚未實現
* 架構稍微整理，重寫一些原始碼，希望方便 bert 版的嘗試

## Environment

* python 3.6
* elmo
* tensorflow == 1.15
* [ckiptagger](https://github.com/ckiplab/ckiptagger)

## Usage

### Prepare

#### ckiptagger

* 請看 [ckiptagger github](https://github.com/ckiplab/ckiptagger)
* 下載模型文件等
* 注意 tensorflow 不可用 2.0 以上版本

#### elmoformanylangs

* 從[這裡](https://github.com/HIT-SCIR/ELMoForManyLangs)下載 elmo model
* 注意 pip install elmoformanylangs 後，要把 `elmo.py class Embedder(object)`
"""py
if output_layer == -1:
     payload = np.average(data, axis=0)
else:
     payload = data[output_layer]
"""
改成
"""py
if output_layer == -1:
     payload = np.average(data, axis=0)
 #code changed here
 elif output_layer == -2:
     payload = data
 else:
     payload = data[output_layer]
"""
* 詳情可看[這裡](https://github.com/HIT-SCIR/ELMoForManyLangs/issues/31)

### Code
```py
from SifRankZh import SIFRank
from ckiptagger import WS, POS
from elmoformanylangs import Embedder


# Load ckiptagger
ws = WS("CKIP_MODEL_FOLDER", disable_cuda=False)
pos = POS('CKIP_MODEL_FOLDER', disable_cuda=False)
# Load elmo
elmo = Embedder('ELMO_MODEL_FOLDER')
sifrank = SIFRank(elmo, ws, pos)

text = ['''王品牛排光復南店從1999年開募到現在，走過了21個年頭，是不少在地人家庭聚餐的回憶，為了紀念老店熄燈，店家也在最後營業的週末，贈送玫瑰花給客人，甚至也發給鄰里其他分店折扣感謝卡，感謝週邊鄰里的支持。讓不少人感慨又一個時代的時代的眼淚落下。這個店面就在光復南路與信義路口附近，距離通化夜市、台北101都不遠，過去遍佈全台的王品牛排，目前在台北、新竹、台中、台南、高雄等地都有分店，現在隨著光復南店熄燈，王品牛排分店總數將降到11間。''', 
'''24小時不打烊的書店「誠品敦南店」，今（31）日將吹熄燈號，書店舉辦18小時不間斷馬拉松講座，PChome董事長詹宏志在凌晨4點30分到場開講，現場人潮擠爆，讓詹宏志驚訝不已，一上台就笑說，台北是一座很神奇的城市。身兼電商董座和作家身分的詹宏志一到場，現場一陣歡呼，台下擠滿人潮，連誠品董事長吳旻潔和總經理李介修也在人海裡頭，不過這個時間，天都還沒亮。'''
]
```

* sifrank.extract_keyphrases

```py
print(sifrank.extract_keyphrases(text, topn=5))
# Output
[[('王品牛排光復南店', 2.5756706856906355), ('分店折扣', 1.2911101382025754), ('週邊鄰里', 1.1250563617118434), ('通化夜市', 0.7099283049352578), ('台的王品', 0.7002897142215967)], [('現場人潮', 2.0921174935663354), ('董事長詹宏志', 1.3693523626207527), ('誠品董事長', 0.7583358332313868), ('電商董座', 0.7203905037136835), ('總經理李介修', 0.6651830981444624)]]
```

* sifrank.extract_keyphrases_with_embedding

```py
sifrank.extract_keyphrases_with_embedding(text, topn=5)
# Return List of Result Object
```

### Result Object

* self.text: 原文
* self.sent_embed: 原文的 embedding
* self[k].keyphrase: 第 k 個 keyphrase
* self[k].score: 第 k 個 keyphrase 的 分數
* self[k].embed: 第 k 個 keyphrase 的 embedding

## Referace

* [SIFRank_zh](https://github.com/sunyilgdx/SIFRank_zh)