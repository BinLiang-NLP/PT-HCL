# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

# from transformers import BertModel
from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils_new_label import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        if self.opt.is_test:
            self.maskset = ABSADataset(opt.dataset_file['mask'], tokenizer)
            self.replaceset = ABSADataset(opt.dataset_file['replace'], tokenizer)
            self.sentenceset = ABSADataset(opt.dataset_file['sentence']+opt.method+'_'+opt.dataset+'.raw', tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

    def run_test(self):
        mask_data_loader = DataLoader(dataset=self.maskset, batch_size=self.opt.batch_size, shuffle=False)
        replace_data_loader = DataLoader(dataset=self.replaceset, batch_size=self.opt.batch_size, shuffle=False)
        sentence_data_loader = DataLoader(dataset=self.sentenceset, batch_size=self.opt.batch_size, shuffle=False)
        orig_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self.model.load_state_dict(torch.load(self.opt.state_dict_path))
        self.model = self.model.to(self.opt.device)

        target_dic = {'dt': 'donald trump', 'hc': 'hillary clinton', 'fm': 'feminist movement', 'la': 'legalization of abortion', 'tp': 'trade policy'}
        testing_target = target_dic[self.opt.cross_dataset]

        w_path = './datasets/'+self.opt.method+'_'+self.opt.dataset
        w_fp = open(w_path, 'w')
        n_cross = 0
        n_not = 0
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for mt_batch, rt_batch, st_batch, o_batch in zip(mask_data_loader, replace_data_loader, sentence_data_loader, orig_data_loader):
                orig_texts = o_batch['text']
                orig_targets = o_batch['target']
                orig_stances = o_batch['polarity'].cpu().detach().numpy()

                mt_inputs = [mt_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                mt_targets = mt_batch['polarity'].to(self.opt.device)
                mt_outputs = self.model(mt_inputs)

                rt_inputs = [rt_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                rt_targets = rt_batch['polarity'].to(self.opt.device)
                rt_outputs = self.model(rt_inputs)
                
                st_inputs = [st_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                st_targets = st_batch['polarity'].to(self.opt.device)
                st_outputs = self.model(st_inputs)
                st_texts = st_batch['text']

                m_predict = torch.argmax(mt_outputs, -1)
                r_predict = torch.argmax(rt_outputs, -1)
                s_predict = torch.argmax(st_outputs, -1)


                for mp, rp, sp, mt, text, target, stance, mask_text in zip(m_predict, r_predict, s_predict, mt_targets, orig_texts, orig_targets, orig_stances, st_texts):
                    
                    if mp == rp and rp == sp and mp == mt:
                        cross_label = '1'
                        n_cross += 1
                    else:
                        cross_label = '0'
                        n_not += 1
                    string = str(text) + '\n' + str(target) + '\n' + str(stance) + '\n' + cross_label + '\n'
                    w_fp.write(string)
        w_fp.close()
        print(self.opt.dataset)
        print(self.opt.method)
        print('cross rate:', n_cross/(n_cross+n_not))
        print('count:', n_cross, n_not)


def main(ds):
    # Hyper Parameters

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'bert_spc': BERT_SPC,
    }
    dataset_files = {
	'dt': {
            'train': './raw_data/dt.raw',
            'test': './raw_data/dt.raw',
            'mask': './augment_data/mask/dt.raw',
            'replace': './augment_data/replace/dt.raw',
            'sentence': './augment_data/sentence/',
        },
        'hc': {
            'train': './raw_data/hc.raw',
            'test': './raw_data/hc.raw',
             'mask': './augment_data/mask/hc.raw',
            'replace': './augment_data/replace/hc.raw',
             'sentence': './augment_data/sentence/',
      },
        'fm': {
            'train': './raw_data/fm.raw',
            'test': './raw_data/fm.raw',
             'mask': './augment_data/mask/fm.raw',
            'replace': './augment_data/replace/fm.raw',
             'sentence': './augment_data/sentence/',
      },
        'la': {
            'train': './raw_data/la.raw',
            'test': './raw_data/la.raw',
             'mask': './augment_data/mask/la.raw',
            'replace': './augment_data/replace/la.raw',
             'sentence': './augment_data/sentence/',
      },
      'a': {
            'train': './raw_data/a.raw',
            'test': './raw_data/a.raw',
             'mask': './augment_data/mask/a.raw',
            'replace': './augment_data/replace/a.raw',
             'sentence': './augment_data/sentence/',
      },
      'cc': {
            'train': './raw_data/cc.raw',
            'test': './raw_data/cc.raw',
             'mask': './augment_data/mask/cc.raw',
            'replace': './augment_data/replace/cc.raw',
             'sentence': './augment_data/sentence/',
      },
        'tp': {
            'train': './raw_data/tp.raw',
            'test': './raw_data/tp.raw',
             'mask': './augment_data/mask/tp.raw',
            'replace': './augment_data/replace/tp.raw',
             'sentence': './augment_data/sentence/',
      },
        'cvs_aet': {
            'train': './raw_data/cvs_aet.raw',
            'test': './raw_data/cvs_aet.raw',
              'mask': './augment_data/mask/cvs_aet.raw',
            'replace': './augment_data/replace/cvs_aet.raw',
             'sentence': './augment_data/sentence/',
     },
        'ci_esrx': {
            'train': './raw_data/ci_esrx.raw',
            'test': './raw_data/ci_esrx.raw',
             'mask': './augment_data/mask/ci_esrx.raw',
            'replace': './augment_data/replace/ci_esrx.raw',
             'sentence': './augment_data/sentence/',
      },
        'antm_ci': {
            'train': './raw_data/antm_ci.raw',
            'test': './raw_data/antm_ci.raw',
             'mask': './augment_data/mask/antm_ci.raw',
            'replace': './augment_data/replace/antm_ci.raw',
             'sentence': './augment_data/sentence/',
      },
        'aet_hum': {
            'train': './raw_data/aet_hum.raw',
            'test': './raw_data/aet_hum.raw',
             'mask': './augment_data/mask/aet_hum.raw',
            'replace': './augment_data/replace/aet_hum.raw',
             'sentence': './augment_data/sentence/',
      },
        'vast': {
            'train': './zeroshot_vast/train.raw',
            'test': './zeroshot_vast/train.raw',
            'mask': './augment_data/mask/vast_mask.raw',
            #'replace': './augment_data/replace/aet_hum.raw',
            'sentence': './augment_data/sentence/',
      },
   }
    input_colses = {
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.dataset=ds

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.is_test:
        print('testing......')
        fpath={
        
            'fm':'state_dict/bert_spc_fm_val_acc_1.0',
            'la':'state_dict/bert_spc_la_val_acc_1.0',
            'hc':'state_dict/bert_spc_hc_val_acc_1.0',
            'dt':'state_dict/bert_spc_dt_val_acc_1.0',
            'a':'state_dict/bert_spc_a_val_acc_1.0',
            'cc':'state_dict/bert_spc_cc_val_acc_1.0',
            #'tp':'state_dict/bert_spc_tp_val_acc_0.9978',

        }
        for aug_method in ['lda']:
            #opt.state_dict_path=fpath[opt.dataset] #put model trained from istest=0 here
            opt.method=aug_method
            ins = Instructor(opt)
            ins.run_test()
    else:
        ins = Instructor(opt)
        ins.run()
        log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
        logger.addHandler(logging.FileHandler(log_file))
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='fm', type=str) #cvs_aet
    parser.add_argument('--cross_dataset', default='hc', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=4, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')

    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--is_test', default=0, type=int)
    parser.add_argument('--state_dict_path', default='', type=str)
    parser.add_argument('--method', default='', type=str)
    
    opt = parser.parse_args()

    main(opt.dataset)
    
