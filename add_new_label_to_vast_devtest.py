#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: add_new_label_to_vast_devtest.py
# @Version: 
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# @For: 
# @Created Time: Sun 04 Jul 2021 02:29:58 AM CST
# ------------------

def process(filename):
    outfilename = filename.split('.pre')[0]+'.raw'
    fin = open(filename, 'r', encoding='utf-8')
    fout = open(outfilename, 'w')
    pre_lines = fin.readlines()
    fin.close()
    for i in range(0, len(pre_lines), 3):
        text = pre_lines[i]
        target = pre_lines[i+1]
        stance = pre_lines[i+2]
        new_label = '0'
        new_data = text + target + stance + new_label + '\n'
        fout.write(new_data)
    fout.close()

if __name__ == '__main__':
    process('./zeroshot_vast/dev.pre')
    process('./zeroshot_vast/test.pre')
