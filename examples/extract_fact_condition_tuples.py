# Copyright (c) 2022 OpenKS Authors, SCIR, HIT. 
# All Rights Reserved.

from openks.models import OpenKSModel
import argparse
# 列出已加载模型
OpenKSModel.list_modules()


parser = argparse.ArgumentParser(description='Implement of SISO, SIMO, MISO, MIMO for Conditional Statement Extraction')

# Model parameters.
parser.add_argument('--train', type=str, default='data/stmts-train.tsv',
					help='location of the labeled training set')
parser.add_argument('--eval', type=str, default='data/stmts-eval.tsv',
					help='location of the evaluation set')
parser.add_argument('--model_name', type=str, default='MIMO_BERT_LSTM',
					help='the model to be trained')
parser.add_argument('--language_model', type=str, default='resources/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default='resources/pubmed-vectors=50.bin',
					help='wordembedding file for words')
parser.add_argument('--out_model', type=str, default='./models/supervised_model',
					help='location of the saved model')
parser.add_argument('--out_file', type=str, default='./results/evaluation_supervised_model',
					help='location of the saved results')
parser.add_argument('--config', type=str, default='',
					help='gates for three input sequence, i.e. LM(gate1, gate2, gate3), POS(gate1, gate2, gate3), CAP(gate1, gate2, gate3)')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--nu_datasets', type=int, default=6)
parser.add_argument('--num_pass', type=int, default=5,
					help='num of pass for evaluation')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--is_semi', action='store_true')
parser.add_argument('--udata', type=str, default='./udata/stmts-demo-unlabeled-pubmed',
					help='location of the unlabeled data')
parser.add_argument('--AR', action='store_true')
parser.add_argument('--TC', action='store_true')
parser.add_argument('--TCDEL', action='store_true')
parser.add_argument('--SH', action='store_true')
parser.add_argument('--DEL', action='store_true')
parser.add_argument('--run_eval', action='store_true')
args = parser.parse_args()

# 算法模型选择配置
platform = 'PyTorch'
executor = 'openie'
model = 'mimo'
print("根据配置，使用 {} 框架，{} 类型的 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
hypernym_discovery = executor(args=args)
hypernym_discovery.run()
print("-----------------------------------------------") 