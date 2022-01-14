
CUDA_VISIBLE_DEVICES=1 python3 train_cl_vast.py --polarities_dim 3 --dataset vast --batch_size 16 --alpha 0.5 --temperatureP 0.07 --temperatureY 0.14 --lr 2e-5

#CUDA_VISIBLE_DEVICES=0 python3 train_aug_wtwt.py --polarities_dim 4 --batch_size 32 --alpha 0.5 --temperatureP 0.07 --temperatureY 0.14 --lr 2e-5

#CUDA_VISIBLE_DEVICES=1 python3 train_aug_sem16.py --polarities_dim 3 --batch_size 32 --alpha 0.5 --temperatureP 0.07 --temperatureY 0.14 --lr 5e-6

#loss需要换成新loss再测一遍就ok了