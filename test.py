from SifRankZh import SIFRank
from ckiptagger import WS, POS
from elmoformanylangs import Embedder
import time

ws = WS("./model/ckip", disable_cuda=False)
pos = POS('./model/ckip', disable_cuda=False)
ELMO = Embedder('model/elmo_tw')
sifrank = SIFRank(ELMO, ws, pos)


def SIFRank_predict(text):
    text = [t.replace('\n', ' ') for t in text]
    keyphrases = sifrank.extract_keyphrases_with_embedding(text, topn=15)
    return keyphrases


text = ['''欸欸
現在開個水族館海鮮餐廳也會被動保譴責了
所以到底怎麼樣動保團體才不會森77啊
就算吃全素了
也不能保證沒有菜蟲保和微生物保
出來維護小小生命的權益
如果這時候有植物保團體跳出來那又要怎麼辦
有冇八卦啦？
''', '''求學時期讀貴族學校讓我學到一個概念
就是所謂富有的外顯是不顧CP值 而真正的內秉意義是看得到價值而非價格
A cynic is a man who knows the price of everything and the value of nothing.
— Oscar Wilde
今天愛莉莎莎 依照她的需求找到符合的房子 儘管房子被評價(somebody)CP值不好
但是對她而言value很高 高於price
這才是一個真正富裕的人的思考境界

不是cynical的窮人能了解''', '''
1.媒體來源:
鏡新聞

2.記者署名:
吳妍

3.完整新聞標題:
罷韓民調比例高「大勢已去」　沈富雄：韓國瑜你就認了

4.完整新聞內文:
隨著罷免投票日期接近，正反雙方頻頻出招攻防，名嘴趙少康日前在政論節目《少康戰情
室》中公布由《TVBS》所做的民調，調查日期從5月18日到5月20日，有效樣本數為1,237
人，趙少康表示，這個樣本數算大的，「最近調查川普的那個民調才做了1,300多人，這
已經做了1,200多個人。」而他們這份民調在95%的信心度誤差小於正負3%。
這份民調顯示，在今年2月7日，也就是總統大選結束後1個月時，一定會去投罷韓投票的
有44%，可能會的有13%，共57%；但到了5月20日，一定會的只剩39%，可能會的則有12%，
不過累積仍有51%。其中，2月時會去投票且投同意罷免的有79%，5月則稍微下滑到73%，
「但你用會去的乘以同意的73%，就占了高雄市民45%」，相當於104萬人，而要將韓國瑜
成功罷免僅需57萬票。
前立委沈富雄表示，從過去歷次來看，《TVBS》民調的母體結構，韓粉的比例相對偏高，
所以這份民調應該對韓是有利的，但看交叉分析的結果，高雄市任何一個行政區，贊成罷
韓的百分比都很高，且有他的一致性，因此他的結論是：「罷韓一定成功。It's over.（
結束了）」，直言「韓國瑜你就認了。」
5.完整新聞連結 (或短網址):
※ 當新聞連結過長時，需提供短網址方便網友點擊
https://www.mirrormedia.mg/story/20200525edi008/

6.備註:
※ 一個人一天只能張貼一則新聞，被刪或自刪也算額度內，超貼者水桶，請注意

  心得：韓總可以準備去選2022的台北市長了，畢竟現任的都說你很有料了

        選上的機率很大啦''', '''北捷表示，今年1至4月合計虧損4.69億元，將依減租優惠標準，向北市府提捷運系統財產租金減半的紓困申請，目前細節規劃中。▲北捷表示，今年1至4月合計虧損4.69億元。（圖／資料照）台北捷運表示，今年1至4月運輸本業虧損新台幣11.07億元，業外（包含附屬事業等）盈餘6.38億，合計虧損4.69億元。之前有跟交通部申請紓困，但收到回覆說權責在地方，因此擬向台北市政府申請租金減半，細節尚在討論。北捷表示，相較去年同期運輸本業僅虧損0.86億元，業外（包含附屬事業等）盈餘7.82億元，還有盈餘6.96億元，顯見武漢肺炎疫情影響很大，今年與去年同期獲利表現，這樣來回就相差約11億元。''']
print("-" * 20)
# print("原文:"+text)
print("-" * 20)
print("SIFRank_zh结果:")
start = time.time()
for result in SIFRank_predict(text):
    print(result[0].keyphrase)

print(time.time() - start)
print(len(text))
