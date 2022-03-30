# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import argparse
from openks.models.pytorch import semeval_constant as constant
from openks.loaders import loader_config, SourceType, FileType, Loader
from openks.models import OpenKSModel

''' 载入数据 '''
# TODO
dataset = None

''' 文本信息抽取模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
# 算法模型选择配置
parser = argparse.ArgumentParser(description='RE args.')
parser.add_argument("--model", default=None, type=str, required=True)
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--eval_per_epoch", default=10, type=int,
                    help="How many times it evaluates on dev set per epoch")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--negative_label", default="no_relation", type=str)
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
parser.add_argument("--eval_with_gold", action="store_true", help="Whether to evaluate the relation model with gold entities provided.")
parser.add_argument("--train_batch_size", default=32, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--eval_metric", default="f1", type=str)
parser.add_argument("--learning_rate", default=None, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda", action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--seed', type=int, default=0,
                    help="random seed for initialization")
parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")
parser.add_argument("--entity_output_dir", type=str, default=None, help="The directory of the prediction files of the entity model")
parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json", help="The entity prediction file of the dev set")
parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json", help="The entity prediction file of the test set")
parser.add_argument("--prediction_file", type=str, default="predictions.json", help="The prediction filename for the relation model")
parser.add_argument("--feature_file", type=str, default="feature_default", help="The prediction filename for the relation model")
parser.add_argument('--task', type=str, default=None, required=True, choices=['ace04', 'ace05', 'scierc'])
parser.add_argument('--context_window', type=int, default=0)
parser.add_argument('--add_new_tokens', action='store_true', 
                    help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                            "0 (default value): dynamic loss scaling.\n"
                            "Positive power of 2: static loss scaling value.\n")
args = parser.parse_args()

platform = 'PyTorch'
executor = 'RelationExtraction'
model = 'RelationExtraction'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
nero = executor(dataset=dataset, model=OpenKSModel.get_module(platform, model), args=args)
nero.run()

print("-----------------------------------------------")
