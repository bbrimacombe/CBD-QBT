#### QBT
For the QBT method we have updated:
- /train.py
- /src/trainer.py
- /src/model/transformer.py
- /src/data/loader.py
- /src/evaluation/evaluator.py  

QBT-Synced:
export NGPU=8; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --exp_name unsupMT_enfr --dump_path ./dumped/ --reload_model checkpoint.enfr.out.e10.pth,checkpoint.enfr.out.e10.pth --data_path ./data/processed/en-fr/ --lgs en-fr --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 1000 --batch_size 16 --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 6000 --eval_bleu true --validation_metrics valid_en-fr_mt_bleu --beam_size 5 --length_penalty 0.6 --bt_steps en-fr-en,fr-en-fr --encoder_bt_steps en-fr-en,fr-en-fr --qbt_steps en-fr-en,fr-en-fr





# Cross-model Back-translated Distillationfor Unsupervised Machine Translation
#### Accepted as conference paper at 38th International Conference on Machine Learning (ICML 2021).
#### Authors: Xuan-Phi Nguyen, Shafiq Joty, Thanh-Tung Nguyen, Wu Kui, Ai Ti Aw

Paper link: [https://arxiv.org/abs/1911.01986](https://arxiv.org/abs/1911.01986)

# Citation

Please cite as:

```bibtex
@incollection{nguyen2021cbd,
title = {Cross-model Back-translated Distillation for Unsupervised Machine Translation},
author = {Xuan-Phi Nguyen and Shafiq Joty and Thanh-Tung Nguyen and Wu Kui and Ai Ti Aw},
booktitle = {38th International Conference on Machine Learning},
year = {2021},
}
```

These guidelines demonstrate the steps to run CBD on the WMT En-De

### Finetuned model

Model | Train Dataset | Finetuned model
---|---|---
`WMT En-Fr` | [WMT English-French](not-ready) | model: [download](https://www.dropbox.com/s/qi02mbeh39cpow8/checkpoint1.infer.pth?dl=0) 
`WMT En-De` | [WMT English-German](not-ready) | model: [download](https://drive.google.com/file/d/1PEH6sW3Vp2RuwLLblJNxUm7L18zHgXhz/view?usp=sharing) 

#### 0. Installation

```bash
./install.sh
pip install fairseq==0.8.0 --progress-bar off
```

#### 1. Prepare data

Following instructions from [MASS-paper](https://github.com/microsoft/MASS) to create WMT En-De dataset.

#### 2. Prepare pretrained model

Download XLM finetuned model (theta_1): [here](https://drive.google.com/file/d/1EiJSwR49fD3N-iBpAsy0jv-18CdOd1sN/view?usp=sharing), save it to bash variable `export xlm_path=...`

Download MASS finetuned model (theta_2): [here](https://modelrelease.blob.core.windows.net/mass/mass_ft_ende_1024.pth), save it to `export mass_path=....`

Download XLM pretrained model (theta): [here](https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth), save it to `export pretrain_path...`


#### 3. Run CBD model
```bash

# you may change the inputs in the file according to your context
bash run_ende.sh

```

