#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: preprocess_csv.py
# @Version: 
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# @For: 
# @Created Time: Fri 02 Jul 2021 07:57:51 PM CST
# ------------------

import csv
from nltk import word_tokenize
from collections import defaultdict

def process(filename):
    target_dict = defaultdict(int)
    fin = open(filename, 'r')
    fout = open(filename.split('.csv')[0]+'.raw', 'w')
    csv_lines = fin.readlines()
    for row in csv_lines:
        sentence, target, stance = row.strip().split('\t')
        sentence = ' '.join(word_tokenize(sentence))
        string = sentence + '\n' + target + '\n' + stance + '\n'
        
        target_dict[target] += 1
    fin.close()
    target_dict = sorted(target_dict.items(), key=lambda a: -a[1])
    for target, count in target_dict:
        string = target + '\t' + str(count) + '\n'
        fout.write(string)
    fout.close()
    print('done !!!', filename)

if __name__ == '__main__':
    #process('./zeroshot_vast/train.csv')
    #process('./zeroshot_vast/dev.csv')
    #process('./zeroshot_vast/test.csv')
    process('./zeroshot_vast/all.csv')
