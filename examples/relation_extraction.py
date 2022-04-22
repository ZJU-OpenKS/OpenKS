import argparse
from openks.models import OpenKSModel
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer

"""
how to run relation_extraction model
cd OpenKS
python -m examples.relation_extraction \
                            --model_name_or_path bert-base-uncased \
                            --device gpu \
                            --seed 42 \
                            --do_train \
                            --data_path ./openks/models/paddle/data_for_relation_extraction \
                            --max_seq_length 128 \
                            --batch_size 8 \
                            --num_train_epochs 12 \
                            --learning_rate 2e-5 \
                            --warmup_ratio 0.06 \
                            --output_dir ./openks/models/paddle/checkpoints
"""
# 列出已加载模型
OpenKSModel.list_modules()

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./openks/models/paddle/data_for_relation_extraction", type=str, required=False, help="Path to data.")
parser.add_argument("--predict_data_file", default="./data/test_data.json", type=str, required=False, help="Path to data.")
parser.add_argument("--output_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")

args = parser.parse_args()

platform = 'Paddle'
executor = 'Relation_Extraction'
model = 'Relation_Extraction'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
hypernym_discovery = executor(args=args)
hypernym_discovery.run()

print("-----------------------------------------------")