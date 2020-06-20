from typing import List, Tuple
import nltk


class NPExtractor:
    def __init__(self, grammar: str = None):
        # default For CKIP
        if grammar is None:
            self.grammar = """  NP:
        {<A>*<Na|Nb|Nc|Ncd|Nes|Nv|Ng|>{1}<Na|Nb|Nc|Ncd|Nes|Nv|Ng>{0,1}} # Adjective(s)(optional) + Noun(s)"""
        self.np_parser = nltk.RegexpParser(self.grammar)

    def extract_list(self, token_tag_list: List[List[Tuple[str]]]):
        candidate_list = [self.extract(token_tag) for token_tag in token_tag_list]
        return candidate_list

    def extract(self, token_tag: List[Tuple[str]]):
        candidate = []
        np_token_tag = self.np_parser.parse(token_tag)
        count = 0
        for token in np_token_tag:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np = ''.join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length
                if len(np) > 1 and not ' ' in np:
                    candidate.append((np, start_end))
            else:
                count += 1
        return candidate
