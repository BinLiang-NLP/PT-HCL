# -*- coding: utf-8 -*-
# file: BERT_PT_HCL.py
# author: bin liang <bin.liang@stu.hit.edu.cn>
# Based on the BERT_SPC.py from songyouwei <youwei0314@gmail.com>.
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT_PT_HCL(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_PT_HCL, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.mlp1 = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.mlp2 = nn.Linear(opt.bert_dim, opt.bert_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs

        text_embed, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        hidden1=pooled_output
        
        # print(text_embed.shape)
        # exit()
        # print(pooled_output.shape)# [16,768]
        # exit()
        logits = self.dense(pooled_output)

        #pooled_output = self.mlp1(self.mlp2(pooled_output))
        pooled_output_feature = pooled_output.unsqueeze(1)
        pooled_output_feature = F.normalize(pooled_output_feature, dim=2)

        # print(pooled_output)
        hidden2=pooled_output_feature
        return logits, pooled_output_feature,hidden1,hidden2
