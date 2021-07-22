from __future__ import unicode_literals
import os
import torch
import argparse
from . import semeval_constant as constant
import json
import random
import numpy as np
import jieba

from .main import nero_run, read


parser = argparse.ArgumentParser(description='NERO args.')


parser.add_argument("--dataset", type=str, default="semeval", help='')
parser.add_argument("--mode", type=str, default="regd", help="pretrain / pseudo / regd")
parser.add_argument("--gpu", type=str, default="1", help="The GPU to run on")


parser.add_argument("--pattern_file", type=str, default="./data/supply_cooperate_20210518_data/yanbao_ic_pattern.json", help="")
parser.add_argument("--target_dir", type=str, default="data", help="")
parser.add_argument("--log_dir", type=str, default="./log/event", help="")
parser.add_argument("--save_dir", type=str, default="./log/model", help="")
parser.add_argument("--word2vec_file", type=str, default="/home/ps/disk_sdb/yyr/codes/NEROtorch/embedding_model", help="")



parser.add_argument("--train_file", type=str, default="./data/supply_cooperate_20210518_data/train.json", help="")
parser.add_argument("--dev_file", type=str, default="./data/supply_cooperate_20210518_data/test.json", help="")
parser.add_argument("--test_file", type=str, default="./data/supply_cooperate_20210518_data/test.json", help="")

parser.add_argument("--emb_dict", type=str, default="./data/supply_cooperate_20210518_data/emb_dict.json", help="")

parser.add_argument("--checkpoint", type=str, default="./checkpoint/model.ckpt", help="")

parser.add_argument("--train_mode", type=str, default="train", help="train or predict")

parser.add_argument("--glove_word_size", type=int, default=int(2.2e6), help="Corpus size for Glove")
parser.add_argument("--glove_dim", type=int, default=300, help="Embedding dimension for Glove")
parser.add_argument("--top_k", type=int, default=100000, help="Finetune top k words in embedding")
parser.add_argument("--length", type=int, default=110, help="Limit length for sentence")
parser.add_argument("--num_class", type=int, default=len(constant.LABEL_TO_ID), help="Number of classes")


parser.add_argument("--gt_batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--pseudo_size", type=int, default=32, help="Batch size for pseudo labeling")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of epochs")
parser.add_argument("--init_lr", type=float, default=0.0001, help="Initial lr")
parser.add_argument("--lr_decay", type=float, default=0.7, help="Decay rate")
parser.add_argument("--keep_prob", type=float, default=0.7, help="Keep prob in dropout")
parser.add_argument("--grad_clip", type=float, default=5.0, help="Global Norm gradient clipping rate")
parser.add_argument("--hidden", type=int, default=150, help="Hidden size")
parser.add_argument("--att_hidden", type=int, default=150, help="Hidden size for attention")

parser.add_argument("--alpha", type=float, default=0.1, help="Weight of pattern RE")
parser.add_argument("--beta", type=float, default=0.2, help="Weight of similarity score")
parser.add_argument("--gamma", type=float, default=0.5, help="Weight of pseudo label")
parser.add_argument("--tau", type=float, default=0.7, help="Weight of tau")
parser.add_argument("--patterns", type=list, default=[], help="pattern list")




def seed_torch(seed=42):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def main():
    config = parser.parse_args()

    with open(config.pattern_file, "r") as fh:
        patterns = json.load(fh)
    print(config)
    config.patterns = patterns
    data = read(config)
    

    nero_run(config, data)
    





if __name__ == "__main__":
    seed_torch()
    main()
