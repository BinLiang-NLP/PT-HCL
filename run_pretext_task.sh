CUDA_VISIBLE_DEVICES=0 python3 train_new_label_for_vast.py --polarities_dim 3 --batch_size 16 --lr 2e-5 --is_test 0
#CUDA_VISIBLE_DEVICES=0 python3 train_new_label_for_vast.py --polarities_dim 3 --batch_size 16 --lr 2e-5 --is_test 1 --state_dict_path 'state_dict/bert_spc_vast_val_acc_0.8678'

#CUDA_VISIBLE_DEVICES=0 python3 train_new_label.py --dataset wtwt --polarities_dim 4 --batch_size 32 --lr 2e-5 --is_test 0
#CUDA_VISIBLE_DEVICES=0 python3 train_new_label.py --dataset cvs_aet --polarities_dim 4 --batch_size 32 --lr 2e-5 --is_test 1 --state_dict_path 'state_dict/bert_spc_cvs_aet_val_acc_0.9905'

#CUDA_VISIBLE_DEVICES=1 python3 train_new_label.py --dataset sem16 --polarities_dim 3 --batch_size 32 --lr 5e-6 --is_test 0
#CUDA_VISIBLE_DEVICES=1 python3 train_new_label.py --dataset fm --polarities_dim 3 --batch_size 32 --lr 5e-6 --is_test 1 --state_dict_path 'state_dict/bert_spc_fm_val_acc_1.0'
