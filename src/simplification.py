import numpy as np
import random
import json
import nltk
#import requests
#import verb
#import inflect
#import plural

from transformers import ElectraTokenizer, ElectraForPreTraining
import torch
import TextFooler.criteria as criteria

class Simplifier:

    def __init__(self, cos_sim_file='cos_sim_counter_fitting.npy',
                 counterfitted_vectors='TextFooler/counter-fitted-vectors.txt',
                 frequency_dict='frequency.json',threshold=0.0,syn_num = 10, syn_threshold = 0.7, ratio = 0.2, most_freq_num = 5):
        self.threshold = threshold
        self.ratio = ratio
        self.cos_sim = np.load(cos_sim_file)
        self.stop_words_set = criteria.get_stopwords()
        with open(frequency_dict) as f:
            self.freq_dict = json.load(f)
        idx2word = {}
        word2idx = {}
        with open('TextFooler/counter-fitted-vectors.txt', 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.syn_num = syn_num
        self.syn_threshold = syn_threshold
        self.most_freq_num = most_freq_num
    
    def perturb_words(self, tokenized, perturb_ratio):
        len_text = len(tokenized)
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        perturb_indexes = []
        perturb_words = []
        for idx in perturb_idxes:
            if tokenized[idx] not in self.stop_words_set:
                perturb_indexes.append(idx)
                perturb_words.append(tokenized[idx])
        return perturb_words, perturb_indexes

    def complex_plus_random_v1(self, tokenized, perturb_ratio):
        complex_indexes = []
        notcomplex_indexes = []
        complex_words = []
        complex_num = 0
        for (i, word) in enumerate(tokenized):
            chosen = 0
            if self.freq_dict.get(word) != None:
                freq = self.freq_dict.get(word)
                if freq < self.threshold and tokenized[i] not in self.stop_words_set:
                    complex_indexes.append(i)
                    complex_num += 1
                    chosen = 1
            if chosen == 0:
                notcomplex_indexes.append(i)
        if complex_num < int(len(tokenized) * perturb_ratio):
            perturb_idxes = random.sample(notcomplex_indexes, int(len(tokenized) * perturb_ratio)-complex_num)
            for idx in perturb_idxes:
                if tokenized[idx] not in self.stop_words_set:
                    complex_indexes.append(idx)
        else:
            complex_indexes = random.sample(complex_indexes, int(len(tokenized) * perturb_ratio))
        for idx in complex_indexes:
            complex_words.append(tokenized[idx])
        '''
        complex_indexes = []
        notcomplex_indexes = []
        complex_words = []
        for (i, word) in enumerate(tokenized):
            chosen = 0
            if self.freq_dict.get(word) != None:
                freq = self.freq_dict.get(word)
                if freq < self.threshold and tokenized[i] not in self.stop_words_set:
                    complex_indexes.append(i)
                    chosen = 1
            if chosen == 0:
                notcomplex_indexes.append(i)
        perturb_idxes = random.sample(notcomplex_indexes, int(len(tokenized) * perturb_ratio))
        for idx in perturb_idxes:
            if tokenized[idx] not in self.stop_words_set:
                complex_indexes.append(idx)
        for idx in complex_indexes:
            complex_words.append(tokenized[idx])
        '''
        return complex_words, complex_indexes

    def perturb_words(self, tokenized, perturb_ratio):
        len_text = len(tokenized)
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        perturb_indexes = []
        perturb_words = []
        for idx in perturb_idxes:
            if tokenized[idx] not in self.stop_words_set:
                perturb_indexes.append(idx)
                perturb_words.append(tokenized[idx])
        return perturb_words, perturb_indexes

    def get_synonyms(self, word, number = 30, threshold = 0.7):
        sim_words = []
        if(self.word2idx.get(word) == None) :
            return []
        word_index = self.word2idx[word]
        sim_order = np.argsort(-self.cos_sim[word_index, :])[1:1 + number]
        sim_values = self.cos_sim[word_index][sim_order]
        for i in range(number):
            if sim_values[i] >= threshold:
                sim_words.append(self.idx2word[sim_order[i]])
        return sim_words

    def sorted_by_frequency(self,words):
        list = []
        for word in words:
            if self.freq_dict.get(word) != None :
                freq = self.freq_dict.get(word)
            else:
                freq = 0
            list.append((word,freq))
        list.sort(key=lambda  x : x[ 1 ],reverse=True)
        return list
    
    def sorted_by_frequency_reverse(self,words):
        list = []
        for word in words:
            if self.freq_dict.get(word) != None :
                freq = self.freq_dict.get(word)
            else:
                freq = 0
            list.append((word,freq))
        list.sort(key=lambda  x : x[ 1 ],reverse=False)
        return list

    def simplify_v2(self,sentence):
        tokenized = nltk.tokenize.word_tokenize(sentence)
        return " ".join(tokenized)

    def random_freq_v1(self, sentence):
        tokenized = nltk.tokenize.word_tokenize(sentence)
        perturbed_sentence = tokenized[:]
        perturb_words, perturb_indexes = self.perturb_words(tokenized, self.ratio)
        for i, word in enumerate(perturb_words):
            synonyms = self.get_synonyms(word, number=self.syn_num)
            if synonyms == []:
                continue
            if len(synonyms) > self.most_freq_num:
                replace_index = random.randint(0, self.most_freq_num)
            else:
                replace_index = random.randint(0, len(synonyms) - 1)
            replaced_word = self.sorted_by_frequency(synonyms)[replace_index][0]
            perturbed_sentence[perturb_indexes[i]] = replaced_word
            # print(word + " is replaced by " +replaced_word)
        return " ".join(perturbed_sentence)

    def random_freq_v2(self, sentence):
        tokenized = nltk.tokenize.word_tokenize(sentence)
        perturbed_sentence = tokenized[:]
        perturb_words, perturb_indexes = self.complex_plus_random_v1(tokenized, self.ratio)
        for i, word in enumerate(perturb_words):
            synonyms = self.get_synonyms(word, number=self.syn_num)
            if self.freq_dict.get(word) != None:
                orig_freq = self.freq_dict.get(word)
            else:
                orig_freq = 0
            if synonyms == []:
                continue
            replaced_freq = 0
            times = 0
            replaced_word = None
            while replaced_freq < self.threshold and replaced_freq < orig_freq and times < 5:
                if len(synonyms) > self.most_freq_num:
                    replace_index = random.randint(0, self.most_freq_num)
                else:
                    replace_index = random.randint(0, len(synonyms) - 1)
                replaced_freq = self.sorted_by_frequency(synonyms)[replace_index][1]
                replaced_word = self.sorted_by_frequency(synonyms)[replace_index][0]
                times += 1
            if times == 5:
                continue
            if replaced_word != None:
                perturbed_sentence[perturb_indexes[i]] = replaced_word
            # print(word + " is replaced by " +replaced_word)
        return " ".join(perturbed_sentence)