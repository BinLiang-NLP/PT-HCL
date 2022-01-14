import torch
from torch.utils.data import Dataset,DataLoader,SequentialSampler
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
import os
import pandas as pd
from bertconfig import base_config
from sklearn.model_selection import KFold

class BertDataset(Dataset):
    def __init__(self, tokenizer,args,mode=0):
        super(BertDataset,self).__init__()
        self.mode = mode
        self.args=args
        self.label2id = {"AGAINST": 0, 'NONE': 1, "FAVOR": 2,"UNRELATED":3}
        self.path = self._get_data_path()
        self.df = self.get_data()
        self.tokenizer = tokenizer
        self.target_index = self._get_target_index()


    def get_data(self):
        data = []
        id2label = {v:k for k,v in self.label2id.items()}
        with open(self.path, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
            lines = file.readlines()
            id = 0
            for i in range(0, len(lines), 3):
                id = id+1
                text = lines[i].lower().strip()
                target = lines[i + 1].lower().strip()
                stance = lines[i + 2].strip()
                stance = id2label[int(stance) + 1]
                data.append({"ID":id,"Target":target,"Tweet":text,"Stance":stance})
                # print(id)
        return pd.DataFrame(data,columns=["ID","Target","Tweet","Stance"])


    def __getitem__(self, index):
        series = self.df.iloc[index, :]
        id = series['ID']
        target = series['Target']
        tweet = series['Tweet']
        stance = series['Stance']
        return self.transform(id, target, tweet, stance)


    def __len__(self):
        return self.df.shape[0]

    def _get_target_index(self):
        targets = self.df.Target.unique()
        target_dict = {}
        for target in targets:
            target_dict[target] = list(self.df[self.df.Target==target].index)
        return target_dict


    def _get_data_path(self):
        if self.mode==0:
            return os.path.join(self.args.data_dir,self.args.train_name)
        if self.mode==1:
            return os.path.join(self.args.data_dir, self.args.dev_name)
        if self.mode==2:
            return os.path.join(self.args.data_dir, self.args.test_name)


    def transform(self,id, target, tweet, stance):
        label = self.label2id[stance]
        sample_id = id
        sentence_tokens = self.tokenizer.tokenize(tweet)
        target_tokens = self.tokenizer.tokenize(target)
        if len(sentence_tokens) + len(target_tokens) > self.args.max_seq_len - 3:
            if len(sentence_tokens) > self.args.max_seq_len - 3:
                sentence_tokens = sentence_tokens[:self.args.max_seq_len - 3 - len(target_tokens)]  ##truncate content
            else:
                sentence_tokens = sentence_tokens[:self.args.max_seq_len - 3 - len(target_tokens)]
        # target_tokens = ["[MASK]"]
        tokens = ["[CLS]"] + target_tokens + ["[SEP]"] +  sentence_tokens+ ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask_ids = [1] * len(input_ids)
        padding_len = self.args.max_seq_len-len(tokens)
        input_ids = input_ids + [0]*padding_len
        segment_ids = [0]*(len(sentence_tokens)+2)+[1]*(len(target_tokens)+1)+[0]*padding_len
        mask_ids += [0]*padding_len
        input_ids = torch.tensor(input_ids)
        segment_ids = torch.tensor(segment_ids)
        mask_ids = torch.tensor(mask_ids)
        label = torch.tensor(label, dtype=torch.long)
        assert input_ids.shape[0]==mask_ids.shape[0]==segment_ids.shape[0]==self.args.max_seq_len, "assert {} {} {} ".format(input_ids.shape[0],mask_ids.shape[0],segment_ids.shape[0])
        return input_ids,mask_ids,segment_ids,label



if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained("/home/fuyonghao/pretrained_models/bert_base_en")
    data_set = BertDataset(tokenizer,base_config,mode=0)

    import numpy as np
    kf = KFold(n_splits=10,shuffle=True,random_state=10)
    print(len(data_set))

    for i in range(5):
        print(data_set[[0,3,5,6]])




