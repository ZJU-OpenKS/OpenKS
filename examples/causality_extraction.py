# -*-coding:utf-8-*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import argparse
from openks.models import OpenKSModel


parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=50, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--train_data", type=str, default='../openks/data/data_for_causality_extraction/train-corpus.json', help="train data")
parser.add_argument("--test_data", type=str, default='../openks/data/data_for_causality_extraction/test-corpus.json', help="test data")

parser.add_argument("--predict_save_path", type=str, default='../openks/data/data_for_causality_extraction/predict.json', help="predict data save path")
parser.add_argument("--predict_data", type=str, default='../openks/data/data_for_causality_extraction/test-corpus.json', help="predict data")
parser.add_argument("--MLP_save_path", type=str, default='checkpoints/MLP', help="predict data")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1,
                    help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default='checkpoints/Erine', help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default='checkpoints',
                    help="already pretraining trigger detection model checkpoint")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")

args = parser.parse_args()

platform = 'Paddle'
executor = 'Causality_Extraction'
model = 'Causality_Extraction'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
Event_Extraction = executor(args=args)
Event_Extraction.run()

print("-----------------------------------------------")

print("-----------------------------------------------")