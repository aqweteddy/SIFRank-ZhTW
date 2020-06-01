import numpy as np
from typing import List, Tuple

class SentEmbedding:
    def __init__(self, embeddor, stopword_set: set, considered_tags: set, freq_weight_file: str):
        self.embeddor = embeddor
        self.min_seq_len = 16
        self.word2weight = self.load_weight(freq_weight_file)
        self.stopword = stopword_set
        self.considered_tags = considered_tags

    def get_sent_np_embedding(self, token_list: List[List[str]], token_tag_list:List[List[Tuple[str, str]]], candidate_list: List[List[Tuple[str, Tuple]]]):
        """batch get sentence embedding and noun phrase embedding

        Arguments:
            token_list {List[List[str]]} -- tokenized words for each sentence (batch, words)
            token_tag_list {List[List[Tuple[str, str]]]} -- (batch, words, (word, pos))
            candidate_list {List[List[Tuple[str, Tuple]]]} -- (batch, np, (word, (start, end)))

        Returns:
            [type] -- [description]
        """
        # token_seg_list = [self.get_sent_segmented(token) for token in token_list]
        embed = self.get_words_embedding(
            token_list)  # [batch_size, layers, seq_len, hidden_size]

        weight_list = [self.get_weight(token) for token in token_list]

        sent_embed_list = [self.get_avg_weight(
            token_list[i], token_tag_list[i], weight_list[i], embed[i]) for i in range(len(token_list))]

        candidate_embed_list = []
        for i in range(len(candidate_list)):  # iterate article
            candidates_embed = []
            for candidate in candidate_list[i]:
                start = candidate[1][0]  # candidate: [text, (start, end)]
                end = candidate[1][1]
                candidates_embed.append(self.get_np_avg_weight(
                    token_list[i], token_tag_list[i], weight_list[i], embed[i], start, end))
            candidate_embed_list.append(candidates_embed)
        return sent_embed_list, candidate_embed_list

    def get_words_embedding(self, sent_sep: List[List[str]]):
        """get embedding for each word

        Arguments:
            sent_sep {List[List[str]]} -- List of sentences of words

        Returns:
            [batch_size(number of sentence), seq_length, 1024] -- embedding
        """
        max_len = max([len(sent) for sent in sent_sep])
        embed = self.embeddor.sents2elmo(sent_sep, output_layer=-2)
        embed = [np.pad(emb, pad_width=(
            (0, 0), (0, max_len-emb.shape[1]), (0, 0)), mode='constant') for emb in embed]
        embed = np.array(embed)
        return embed

    def get_np_avg_weight(self, token, token_tag, sent_weight, embed, start, end):
        """get noun phrase avg. weights using SIF

        Arguments:
            token {List[str]} -- tokenized sentence
            token_tag {List[Tuple[str]]} -- tokenized word, pos
            sent_weight {List[float]} -- frequncy weight
            embed {np.array.shape(layers, seq_length)} -- sentence embedding
            start {int} -- start position of the noun phrase in index of token
            end {int} -- end position of the noun phrase in index of token

        Returns:
            [num_layer, 1024] -- average weight of noun phrase
        """
        avg = np.zeros((3, 1024))
        length = len(token)

        for i in range(3):
            for j in range(start, end):
                if token_tag[j][1] in self.considered_tags:  # POS
                    avg[i] += embed[i][j] * sent_weight[j]  # [1024] * [1]
            avg[i] = avg[i] / float(length)
        return avg

    def get_avg_weight(self, token, token_tag, sent_weight, embed):
        """get sentence avg. weights using SIF

        Arguments:
            token {List[str]} -- tokenized sentence
            token_tag {List[Tuple[str]]} -- tokenized word, pos
            sent_weight {List[float]} -- frequncy weight
            embed {np.array.shape(layers, seq_length)} -- sentence embedding

        Returns:
            [num_layer, 1024] -- average weight of token
        """
        avg = np.zeros((3, 1024))
        length = len(token)

        for i in range(3):
            for j in range(length):
                if token_tag[j][1] in self.considered_tags:  # POS
                    avg[i] += embed[i][j] * sent_weight[j]  # [1024] * [1]
            avg[i] = avg[i] / float(length)
        return avg

    def get_weight(self, token):
        """get sentence freqency weight

        Arguments:
            token {List[str]} -- tokenized sentence

        Returns:
            List[float] -- list of weight (length: len(token))
        """
        weight_list = []
        for word in token:
            if word in self.word2weight.keys():
                weight = self.word2weight[word]
            elif word in self.stopword:
                weight = 0.0
            else:  # OOV
                weight = 0.0
                for w in token:
                    weight = max(weight, self.word2weight.get(w, -1))
            weight_list.append(weight)
        return weight_list

    def load_weight(self, file, weightpara=2.7e-4):
        """load frquency weight from file

        Arguments:
            file {str} -- for each line contain
                                WORD FREQ
        Keyword Arguments:
            weightpara {float} -- OOV word default weight (default: {2.7e-4})

        Returns:
            dict[word] = weight
        """
        with open(file, 'r') as f:
            lines = f.readlines()
        sum_freq = 0
        word2freq = dict()
        for line in lines:
            word_freq = line.split()
            word2freq[word_freq[0]] = float(word_freq[1])
            sum_freq += float(word_freq[1])
        for key, val in word2freq.items():
            word2freq[key] = weightpara / \
                (weightpara + val / sum_freq)  # word2weight
        return word2freq

    @staticmethod
    def align_embedding(embed, token_seg):
        pass

    def get_sent_segmented(self, token):
        sent_sec = []

        if len(token) < self.min_seq_len:
            sent_sec.append(token)
        else:
            pos = 0
            for i, t in enumerate(token):
                if t.find('.') or t.find('。'):
                    if i - pos >= self.min_seq_len:
                        pos += 1
                        sent_sec.append(token[pos: i+1])
            if len(token[pos:]) > 0:
                sent_sec.append(token[pos:])
        return sent_sec
