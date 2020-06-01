from SifRankZh import SIFRank
from ckiptagger import WS, POS
from elmoformanylangs import Embedder
import time

ws = WS("./model/ckip", disable_cuda=False)
pos = POS('./model/ckip', disable_cuda=False)
ELMO = Embedder('model/elmo_tw')
sifrank = SIFRank(ELMO, ws, pos)

text = ['''王品牛排光復南店從1999年開募到現在，走過了21個年頭，是不少在地人家庭聚餐的回憶，為了紀念老店熄燈，店家也在最後營業的週末，贈送玫瑰花給客人，甚至也發給鄰里其他分店折扣感謝卡，感謝週邊鄰里的支持。讓不少人感慨又一個時代的時代的眼淚落下。這個店面就在光復南路與信義路口附近，距離通化夜市、台北101都不遠，過去遍佈全台的王品牛排，目前在台北、新竹、台中、台南、高雄等地都有分店，現在隨著光復南店熄燈，王品牛排分店總數將降到11間。''', 
'''24小時不打烊的書店「誠品敦南店」，今（31）日將吹熄燈號，書店舉辦18小時不間斷馬拉松講座，PChome董事長詹宏志在凌晨4點30分到場開講，現場人潮擠爆，讓詹宏志驚訝不已，一上台就笑說，台北是一座很神奇的城市。身兼電商董座和作家身分的詹宏志一到場，現場一陣歡呼，台下擠滿人潮，連誠品董事長吳旻潔和總經理李介修也在人海裡頭，不過這個時間，天都還沒亮。'''
]
print(sifrank.extract_keyphrases(text))
