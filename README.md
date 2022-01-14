# PT-HCL for Zero-Shot Stance Detection
**The code of this repository is constantly being updated...** 

**Please look forward to it!**

# Introduction
This repository was used in our paper:  

**Zero-Shot Stance Detection via Contrastive Learning**
<br>
Bin Liang, Zixiao Chen, Lin Gui, Yulan He, Min Yang, Ruifeng Xu<sup>\*</sup>. *Proceedings of WWW 2022*

Please cite our paper and kindly give a star for this repository if you use this code.

## Requirements
- pytorch >= 0.4.0
- numpy >= 1.13.3
- sklearn
- python 3.6 / 3.7
- transformers


## Training
* Train with command, optional arguments could be found in [train_pt_hcl_vast.py](/train_pt_hcl_vast.py) \& [train_pt_hcl_sem16.py](/train_pt_hcl_sem16.py) \& [train_pt_hcl_wtwt.py](/train_pt_hcl_wtwt.py)

* Run VAST dataset: ```./run_zssd_vast.sh```

* Run WTWT dataset: ```./run_zssd_wtwt.sh```

* Run SEM16 dataset: ```./run_zssd_sem16.sh```
