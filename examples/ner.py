import argparse
from openks.models import OpenKSModel
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer

"""
how to run ner model
cd OpenKS
python -m examples.ner \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./openks/models/paddle/tmp/msra_ner/ \
    --device gpu
"""
# 列出已加载模型
OpenKSModel.list_modules()

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
parser.add_argument("--init_checkpoint_path", default=None, type=str, help="The model checkpoint path.", )
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")

args = parser.parse_args()

platform = 'Paddle'
executor = 'Ner'
model = 'Ner'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
hypernym_discovery = executor(args=args)
hypernym_discovery.run()

print("-----------------------------------------------")