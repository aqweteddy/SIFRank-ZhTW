from typing import List, Tuple

import numpy as np

from .np_extractor import NPExtractor
from .return_object import Result
from .sent_embedding import SentEmbedding


class SIFRank:
    def __init__(self,
                 pretrained_model,
                 ws,
                 pos,
                 np_grammar=None,
                 stopword_set=set(),
                 considered_tags={'Na', 'Nb', 'Nc',
                                  'Ncd', 'Nes', 'Ng', 'Nv', 'DE', 'A'},
                 word_freq_file='./model/dict.txt'
                 ):
        self.word_embed = pretrained_model
        self.sent_embed = SentEmbedding(self.word_embed,
                                        stopword_set=stopword_set,
                                        considered_tags=considered_tags,
                                        freq_weight_file=word_freq_file)
        self.np_extractor = NPExtractor(np_grammar)
        self.ws = ws
        self.pos = pos

        self.stopword = stopword_set

    def extract_keyphrases(self, 
                           text: List[str], 
                           topn: int = 5
                           ) -> List[List[Tuple[str, str]]]:
        """extract keyphrase

        Arguments:
            text {List[str]} -- articles

        Keyword Arguments:
            topn {int} -- extract topn keyphrases (default: {5})

        Returns:
            List[List[Tuple[str, str]]] -- [[(keyphrase, pos), ...] , ...]
        """
        token_list, token_tag_list = self.get_token_pos(text)
        candidate_list = self.np_extractor.extract_list(token_tag_list)
        sent_embed_list, candidate_embed_list = self.sent_embed.get_sent_np_embedding(
            token_list, token_tag_list, candidate_list)

        results = []
        for sent_embed, candidates_embed, candidates in zip(sent_embed_list, candidate_embed_list, candidate_list):
            candidates_score = []
            for cand_embed in candidates_embed:
                score = self.get_cos_dist(sent_embed, cand_embed)
                candidates_score.append(score)

            result = self.reduce_repeat(candidates, candidates_score)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            results.append(result[:topn])
        return results

    def extract_keyphrases_with_embedding(self, text: List[str], topn=5) -> List[Result]:
        """extract keyphrase with Result Object

        Arguments:
            text {List[str]} -- list of articles

        Keyword Arguments:
            topn {int} -- keep topn keyphrase (default: {5})

        Returns:
            List[Result] -- List of Result objects
        """
        token_list, token_tag_list = self.get_token_pos(text)
        candidate_list = self.np_extractor.extract_list(token_tag_list)
        sent_embed_list, candidate_embed_list = self.sent_embed.get_sent_np_embedding(
            token_list, token_tag_list, candidate_list)

        results = []
        for i in range(len(sent_embed_list)):
            sent_embed, candidates_embed, candidates = sent_embed_list[
                i], candidate_embed_list[i], candidate_list[i]
            candidates_score = []
            result = Result(text=text, sent_embed=sent_embed)

            for cand, cand_embed in zip(candidates, candidates_embed):
                score = self.get_cos_dist(sent_embed, cand_embed)
                result.add_keyphrase(cand[0], cand_embed, score)

            result.reduce_repeat()
            result.sort_kp()
            results.append(result)
        return results

    #! args.method not in SIFRank()
    def reduce_repeat(self, candidates, scores, method='avg'):
        def find_sub_str(keys, word):
            for key in keys:
                if word != key and word in key:
                    return key  # found
            return None

        cand_dict = {}
        for cand, score in zip(candidates, scores):
            if not cand in cand_dict.keys():
                cand_dict[cand[0]] = []
            cand_dict[cand[0]].append(score)

        for cand in cand_dict.keys():
            key = find_sub_str(cand_dict.keys(), cand)
            if key:
                cand_dict[key] += cand_dict[cand]
                cand_dict[cand] = 0.

        for cand, score_list in cand_dict.items():
            if isinstance(cand_dict[cand], int) or not score_list:
                continue

            if method == 'avg':
                score = sum(
                    score_list) if not cand in self.stopword else 0.0
                cand_dict[cand] = score

            elif method == 'max':
                score = max(
                    score_list) if not cand in self.stopword else 0.0
                cand_dict[cand] = score
        return list(cand_dict.items())

    #! args layer_weight not in SIFRank()
    def get_cos_dist(self, sent_embed, kp_embed, layer_weight=[0, 1, 0]):
        score = 0
        for i in range(3):
            a, b = sent_embed[i], kp_embed[i]
            cos_sim = a@b.T / (np.linalg.norm(a) * np.linalg.norm(b))
            score += cos_sim * layer_weight[i]
        return score

    def get_token_pos(self, text: List[str]) -> (List[List[str]], List[List[Tuple[str]]]):
        """get tokenized word and pos

        Arguments:
            List {[str]} -- articles

        Returns:
            [[(word, pos), ...], ...]
        """
        sent_word = self.ws(text)
        sent_tag = self.pos(sent_word)
        token_tag = [list(map(lambda x, y: (x, y), words, tags))
                     for words, tags in zip(sent_word, sent_tag)]
        return sent_word, token_tag

