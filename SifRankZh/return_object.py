from typing import List
import numpy as np

class Result:
    def __init__(self, text, sent_embed):
        self.text = text
        self.sent_embed = sent_embed
        self.kp = []

    def add_keyphrase(self, kp, embed, score):
        self.kp.append(Keyphrase(kp, embed, score))

    def reduce_repeat(self, method='avg'):
        def find_sub_str(keys, word):
            for key in keys:
                if word != key and word in key:
                    return key  # found
            return None

        all_candidates = [k.keyphrase for k in self.kp]
        # remove repeat
        new_kp = []
        for i in range(len(self.kp)):
            a = self.kp[i]
            if a.score == 0:
                continue
            for j in range(i+1, len(self.kp)):
                b = self.kp[j]
                if a.keyphrase == b.keyphrase:
                    a.score += b.score
                    b.score = 0
            new_kp.append(a)
        # remove subword
        # new_kp = []
        # for k in len(self.kp):
        #     if not find_sub_str(all_candidates, k.keyphrase):
        #         new_kp.append(k)
        self.kp = new_kp

    def sort_kp(self):
        self.kp = sorted(self.kp, key=lambda x: x.score, reverse=True)

    def __getitem__(self, k):
        return self.kp[k]


class Keyphrase:
    keyphrase: str
    embed: List
    score: float

    def __init__(self, kp, embed, score):
        self.keyphrase = kp
        self.embed = np.array(embed)
        self.score = score
