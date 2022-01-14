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
import json
from nltk import word_tokenize

def process(filename):
    fin = open(filename, 'r')
    fout = open(filename.split('.csv')[0]+'.raw', 'w')
    csv_lines = csv.DictReader(fin)
    for row in csv_lines:
        text = json.loads(row['text'])
        target = json.loads(row['topic'])
        stance = row['label']
        text_s = []
        for item in text:
            text_s.append(' '.join(item))
        text_s = ' '.join(text_s)
        target_s = ' '.join(target)
        string = text_s + '\n' + target_s + '\n' + str(stance) + '\n'
        fout.write(string)
    fin.close()
    fout.close()
    print('done !!!', filename)

if __name__ == '__main__':
    process('./vast_data/vast_train.csv')
    process('./vast_data/vast_dev.csv')
    process('./vast_data/vast_test.csv')
