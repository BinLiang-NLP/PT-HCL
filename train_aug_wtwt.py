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

#from infoNCE import InfoNCE
from criterion import InfoNCE, Trans_stance

from sklearn import metrics
from time import strftime, localtime

# from transformers import BertModel
from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import BERT_SPC_CL

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

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        elif 'vast' in opt.dataset:
            logger.info('using dev for vast!')
            self.valset = ABSADataset(opt.dataset_file['dev'], tokenizer)
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

    def _train(self, criterion, con_criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_f1_ma=0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total, sub_loss1, sub_loss2, sub_loss3 = 0, 0, 0, 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, logits,hidden1,hidden2 = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)
                cross_labels = batch['cross_label'].to(self.opt.device)

                sup_loss = criterion(outputs, targets)
                cross_loss,al_loss,un_loss = con_criterion(logits, cross_labels, targets)
                loss = self.opt.sup_loss_weight * sup_loss + self.opt.con_loss_weight_cross * cross_loss
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                sub_loss1 += sup_loss.item() * len(outputs)
                sub_loss2 += cross_loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    train_sub_loss1 = sub_loss1 / n_total
                    train_sub_loss2 = sub_loss2 / n_total
                    logger.info('loss: {:.4f}, supervised_loss: {:.4f}, cross_loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_sub_loss1, train_sub_loss2, train_acc))

            val_acc, val_f1, val_f1_ma, val_f1_avg = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}, val_f1_ma: {:.4f}, val_f1_avg: {:.4f}'.format(val_acc, val_f1, val_f1_ma, val_f1_avg))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
            #if val_f1 > max_val_f1: 
            #   max_val_f1 = val_f1
            if val_f1_ma > max_val_f1_ma:
                max_val_f1_ma = val_f1_ma
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict_cl'):
                    os.mkdir('state_dict_cl')
                path = 'state_dict_cl/{0}_{1}_val_f1_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_f1, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

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
                t_outputs, _ ,hidden1,hidden2 = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
        f1_ma = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 2], average='macro')
        f1_mi = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 2], average='micro')
        f1_avg = 0.5 * (f1_ma + f1_mi)
        return acc, f1, f1_ma, f1_avg

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        con_criterion = Trans_stance(self.opt)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, con_criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1, test_f1_ma, test_f1_avg = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, test_f1_ma: {:.4f}, test_f1_avg: {:.4f}'.format(test_acc, test_f1, test_f1_ma, test_f1_avg))
        f_out = open('log/' +self.opt.newname+'_'+str(self.opt.con_loss_weight_cross)+'_'+str(self.opt.sup_loss_weight)+'_'+ self.opt.dataset+'_' + str(self.opt.seed)+'_' + '.txt', 'w', encoding='utf-8')
        f_out.write('test_acc: {:.4f}, test_f1: {:.4f}, test_f1_ma: {:.4f}, test_f1_avg: {:.4f}'.format(test_acc, test_f1, test_f1_ma, test_f1_avg)+'\n')
        return test_f1_ma,test_f1_avg 

def main(ds,method):
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc_cl', type=str)
    parser.add_argument('--dataset', default='cvs_aet', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=50, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=4, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')

    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--con_loss_weight_cross', default=1, type=float)
    parser.add_argument('--sup_loss_weight', default=0.5, type=float)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--temperatureP', default=0.07, type=float)
    parser.add_argument('--temperatureY', default=0.14, type=float)
    parser.add_argument('--alpha', default=0.5, type=float,required=False)
    parser.add_argument('--newname', default='', type=str)
    parser.add_argument('--method', default='', type=str)
    
    opt = parser.parse_args()

    opt.dataset=ds
    opt.method=method

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'bert_spc_cl': BERT_SPC_CL,
    }
    dataset_files = {
	'dt_hc': {
            'train': './datasets/'+opt.method+'_'+'dt',
            'test': './datasets/hc',
        },
        'hc_dt': {
            'train': './datasets/'+opt.method+'_'+'hc',
            'test': './datasets/dt',
      },
        'fm_la': {
            'train': './datasets/'+opt.method+'_'+'fm',
            'test': './datasets/la',
      },
        'la_fm': {
            'train': './datasets/'+opt.method+'_'+'la',
            'test': './datasets/fm',
      },
        'tp_dt': {
            'train': './datasets/tp',
            'test': './datasets/dt',
      },
        'tp_hc': {
            'train': './datasets/tp',
            'test': './datasets/hc',
      },
        'dt_tp': {
            'train': './datasets/dt',
            'test': './datasets/tp',
      },
        'hc_tp': {
            'train': './datasets/hc',
            'test': './datasets/tp',
      },

        'cvs_aet': {
            'train': './datasets/o_'+opt.method+'_'+opt.dataset,
            'test': './datasets/cvs_aet',
     },
        'ci_esrx': {
            'train': './datasets/o_'+opt.method+'_'+opt.dataset,
            'test': './datasets/ci_esrx',
      },
        'antm_ci': {
            'train': './datasets/o_'+opt.method+'_'+opt.dataset,
            'test': './datasets/antm_ci',
      },
        'aet_hum': {
            'train': './datasets/o_'+opt.method+'_'+opt.dataset,
            'test': './datasets/aet_hum',
      },
        'vast': {
            'train': './zeroshot_vast/vast_train.raw',
            'dev': './zeroshot_vast/vast_dev.raw',
            'test': './zeroshot_vast/vast_test.raw',
      },
   }
    input_colses = {
        'bert_spc_cl': ['concat_bert_indices', 'concat_segments_indices'],

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


    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    max_f1_ma=0
    f_max = open('log/max_' +opt.method+'_'+str(opt.con_loss_weight_cross)+'_'+str(opt.sup_loss_weight)+'_'+ opt.dataset+'_' + '.txt', 'a+', encoding='utf-8')
    f_max.close() 
    f_max = open('log/max_' +opt.method+'_'+str(opt.con_loss_weight_cross)+'_'+str(opt.sup_loss_weight)+'_'+ opt.dataset+'_' + '.txt', 'a+', encoding='utf-8')
       
   
    opt.newname='temperatureYP_'+str(opt.alpha)+'_'+str(opt.temperatureP)+'_'+str(opt.temperatureY)+'_'+opt.method
    ins = Instructor(opt)
    now_f1_ma,now_f1_avg=ins.run()
    print(now_f1_ma,',',now_f1_avg)
        
    f_max.close()



if __name__ == '__main__':
    for ds in ['cvs_aet']:#,'ci_esrx','antm_ci','aet_hum'
        for method in ['lda']:
            main(ds,method)
