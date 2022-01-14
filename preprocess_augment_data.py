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

import random

replace_dic = {'dt': ['hc', 'tp'],
               'hc': ['dt', 'tp'],
               'fm': ['la'],
               'la': ['fm'],
               'tp': ['dt', 'hc'],
               'aet_hum': ['antm_ci', 'ci_esrx', 'cvs_aet'],
               'antm_ci': ['aet_hum', 'ci_esrx', 'cvs_aet'],
               'ci_esrx': ['aet_hum', 'antm_ci', 'cvs_aet'],
               'cvs_aet': ['aet_hum', 'antm_ci', 'ci_esrx'],
}

mask_dir = './augment_data/mask/'
replace_dir = './augment_data/replace/'
sentence_dir = './augment_data/sentence/'

def load_seed_words(dataname, percent=1):
    path = './seed_words/'+dataname+'.seed'
    print(path)
    seed_words = {}
    fin = open(path)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        word, weight = line.split()
        if float(weight) <= 0:
            continue
        seed_words[word] = float(weight)
    fin.close()
    save_len = int(len(seed_words) * percent)
    seed_words = sorted(seed_words.items(), key=lambda a: -a[1])
    word_dic = {}
    for word, weight in seed_words:
        word_dic[word] = weight
        save_len -= 1
    return word_dic


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout_mask = open(mask_dir+filename.split('/')[-1], 'w', encoding='utf-8')
    fout_replace = open(replace_dir+filename.split('/')[-1], 'w', encoding='utf-8')
    fout_sentence = open(sentence_dir+filename.split('/')[-1], 'w', encoding='utf-8')
    dataname = filename.split('/')[-1].split('.raw')[0].strip().lower()
    lines = fin.readlines()
    fin.close()
    word_dic = load_seed_words(dataname)
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        target = lines[i+1].lower().strip()
        stance = lines[i+2].lower().strip()
        # deriving masked data
        mask_string = text + '\n' + '[MASK]' + '\n' + stance + '\n'
        # deriving replaced data
        random_id = random.randint(0,len(replace_dic[dataname])-1)
        replace_target = replace_dic[dataname][random_id]
        replace_string = text + '\n' + replace_target + '\n' + stance + '\n'
        # deriving masked sentence's data
        text_list = text.split()
        sentence = []
        for token in text_list:
            if token in word_dic:
                sentence.append('[MASK]')
            else:
                sentence.append(token)
        sentence_string = ' '.join(sentence) + '\n' + target + '\n' + stance + '\n'
        # saving data
        fout_mask.write(mask_string)
        fout_replace.write(replace_string)
        fout_sentence.write(sentence_string)
    fout_mask.close()
    fout_replace.close()
    fout_sentence.close()



if __name__ == '__main__':
    #process('./raw_data/dt.raw')
    #process('./raw_data/fm.raw')
    #process('./raw_data/hc.raw')
    #process('./raw_data/la.raw')
    process("./raw_data/tp.raw")
    #process('./raw_data/aet_hum.raw')
    #process('./raw_data/antm_ci.raw')
    #process('./raw_data/ci_esrx.raw')
    #process('./raw_data/cvs_aet.raw')

