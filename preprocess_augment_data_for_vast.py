#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: preprocess_data.py
# @Version: 
# @Author: BinLiang
# @Mail: 18b951033@stu.hit.edu.cn
# @For: 
# @Created Time: Mon 10 Aug 2020 03:15:03 PM CST
# ------------------

import numpy as np
import string
import nltk
import math
import random
import time
from nltk.corpus import wordnet as wn
from itertools import chain
import spacy

nlp = spacy.load('en_core_web_sm')

import gensim
from sklearn.datasets import fetch_20newsgroups
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
import os
from pprint import pprint
def load_target_seeds():
    path = './seed_words/vast/seeds.txt'
    fin = open(path, 'r', encoding='utf-8')
    target_seeds = {}
    seed_lines = fin.readlines()
    fin.close()
    for i in range(0, len(seed_lines), 2):
        target = seed_lines[i].strip().lower()
        seed_words = seed_lines[i+1].strip().split('\t')
        target_seeds[target] = {}
        count = 0
        for item in seed_words:
            if count > 10:
                continue
            word, weight = item.split(' ')
            target_seeds[target][word] = weight
            count += 1
    return target_seeds


mask_path = './augment_data/mask/vast_mask.raw'
sentence_path = './augment_data/sentence/'

def tokenize(text):
    """
    将text分词，并去掉停用词。STOPWORDS -是指Stone, Denis, Kwantes(2010)的stopwords集合.
    :param text:需要处理的文本
    :return:去掉停用词后的"词"序列
    """
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def process(method,filename):
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout_mask = open(mask_path, 'w', encoding='utf-8')
    fout_sentence = open(sentence_path+method+'_vast.raw', 'w', encoding='utf-8')
    dataname = 'vast'
    lines = fin.readlines()
    fin.close()
    target_seeds = load_target_seeds()

    fin2 = open(filename, 'r', encoding='utf-8', errors='ignore')
    lines2 = fin2.readlines()
    fin2.close()
    replace_dict={}

    for i in range(0, len(lines2), 3):

        if 'dp' in method:
            text=lines[i].lower().strip()
            text2=' '.join(nltk.word_tokenize(text)).lower()
            doc=nlp(text2)
            for token in doc:
                if token.tag_ not in replace_dict:
                    replace_dict[token.tag_]=[token.text]
                else:
                    replace_dict[token.tag_].append(token.text)
    if method=='lda':
        vt_dict={}
        target_topic_word_dict={}
        fin3 = open(filename, 'r', encoding='utf-8', errors='ignore')
        lines3 = fin3.readlines()
        fin3.close()
        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            target = lines[i+1].lower().strip()
            if target not in vt_dict:
                vt_dict[target]=[text]
            else:
                vt_dict[target].append(text)
        f_look=open(filename+"_topic_words__top20", 'w', encoding='utf-8', errors='ignore')
    
        for target in vt_dict:
            processed_docs = [tokenize(text) for text in vt_dict[target]]
            word_count_dict = Dictionary(processed_docs)
            bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
         
            try:
                lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=1, id2word=word_count_dict, passes=20)
            
            except:
                target_topic_word_dict[target]=[]
                continue
            top_topics = lda_model.top_topics(bag_of_words_corpus)
            lda_words=[]
            topk=20
            for x in top_topics:
                for i in range(topk):
                    if i <len(x[0]):
                       
                        lda_words.append(x[0][i][1])
            target_topic_word_dict[target]=lda_words
            
            print(target,':',lda_words)
            f_look.write(target+':'+str(lda_words)+'\n')
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        target = lines[i+1].lower().strip()
        stance = lines[i+2].lower().strip()
         
        mask_string = text + '\n' + '[MASK]' + '\n' + stance + '\n'
           
        text_list = text.split()
        sentence = []
        if method=='dp':
            text2=' '.join(nltk.word_tokenize(text)).lower()
            doc=nlp(text2)
            for token in doc:
                if token.text in target_seeds:
                    sentence.append('['+token.tag_+']')
                else:
                    sentence.append(token.text)
        else:
            for token in text_list:
                if method=='lda':
                    if token in target_topic_word_dict[target]:
                        sentence.append('[LDA_MASK]')
                    else:
                        sentence.append(token)
                    continue    
                if token in target_seeds:
                    if method=='mask':
                        sentence.append('[MASK]')
                    if method=='random_word':
                        n_text_list=text_list[:]
                        for token in text_list:
                            if token in target_seeds:
                                n_text_list.remove(token)
                        random_word=n_text_list[random.randint(0,len(n_text_list)-1)]
                        sentence.append('['+random_word+']')
                    if method=='spelling':
                        randlist=np.random.rand(len(token))
                        count=0
                        random_rate=0.4
                        spelling_word=''
                        for ch in token:
                            if random_rate>randlist[count]:
                                s = string.ascii_letters
                                spelling_word+=random.choice(s).lower()
                            else:
                                spelling_word+=ch
                            count+=1
                        sentence.append('['+spelling_word+']')
                    if method=='delete':
                        continue
                else:
                    sentence.append(token)
        sentence_string = ' '.join(sentence) + '\n' + target + '\n' + stance + '\n'
            # saving data
        fout_mask.write(mask_string)
        fout_sentence.write(sentence_string)
    fout_mask.close()
    fout_sentence.close()



if __name__ == '__main__':
    for method in ['lda']:
        process(method,"./zeroshot_vast/train.raw")


