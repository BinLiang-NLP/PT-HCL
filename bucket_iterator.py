# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text = []
        batch_target = []
        batch_text_indices = []
        batch_target_indices = []
        batch_stance = []
        batch_in_graph = []
        batch_cross_graph = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text, target, text_indices, target_indices, stance, in_graph, cross_graph = \
                item['text'], item['target'], item['text_indices'], item['target_indices'],\
                item['stance'], item['in_graph'], item['cross_graph']
            text_padding = [0] * (max_len - len(text_indices))
            target_padding = [0] * (max_len - len(target_indices))

            batch_text.append(text)
            batch_target.append(target)
            batch_text_indices.append(text_indices + text_padding)
            batch_target_indices.append(target_indices + target_padding)
            batch_stance.append(stance)
            batch_in_graph.append(numpy.pad(in_graph, 
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_cross_graph.append(numpy.pad(cross_graph, 
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))


        return { \
                'text': batch_text, \
                'target': batch_target, \
                'text_indices': torch.tensor(batch_text_indices), \
                'target_indices': torch.tensor(batch_target_indices), \
                'stance': torch.tensor(batch_stance), \
                'in_graph': torch.tensor(batch_in_graph), \
                'cross_graph': torch.tensor(batch_cross_graph),
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]


class BertBucketIterator(BucketIterator):
    def __init__(self,*args,**kwargs):
        super(BertBucketIterator,self).__init__(*args,**kwargs)

    def pad_data(self, batch_data):
        batch_text = []
        batch_target = []
        batch_text_indices = []
        batch_target_indices = []
        batch_stance = []
        batch_in_graph = []
        batch_cross_graph = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text, target, text_indices, attention_mask, stance, in_graph, cross_graph = \
                item['text'], item['target'], item['text_indices'], item['attention_mask'],\
                item['stance'], item['in_graph'], item['cross_graph']
            text_padding = [0] * (max_len - len(text_indices))
            # target_padding = [0] * (max_len - len(target_indices))
            attention_padding = [0] * (max_len - len(text_indices))


            batch_text.append(text)
            batch_target.append(target)
            batch_text_indices.append(text_indices + text_padding)
            batch_target_indices.append(attention_mask + attention_padding)
            batch_stance.append(stance)
            batch_in_graph.append(numpy.pad(in_graph,
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_cross_graph.append(numpy.pad(cross_graph,
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))


        return { \
                'text': batch_text, \
                'target': batch_target, \
                'text_indices': torch.tensor(batch_text_indices), \
                'attention_mask': torch.tensor(batch_target_indices), \
                'stance': torch.tensor(batch_stance), \
                'in_graph': torch.tensor(batch_in_graph), \
                'cross_graph': torch.tensor(batch_cross_graph),
            }
