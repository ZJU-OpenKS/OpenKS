import argparse
from openks.models import OpenKSModel

"""
how to run hypernym detection model
cd OpenKS
python -m examples.hypernym_discovery --mode do_train --train_path YOUR_TRAIN_PATH --dev_path YOUR_DEV_PATH --test_path YOUR_TEST_PATH --save_dir YOUR_CHEKPORINTS_DIR
"""
# 列出已加载模型
OpenKSModel.list_modules()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="do_train", type=str, help="train or eval.")
parser.add_argument("--train_path", default="./dataset/train_hyper.tsv", type=str,
                    help="The train dataset path.")
parser.add_argument("--dev_path", default="./dataset/dev_hyper.tsv", type=str, help="The dev dataset path.")
parser.add_argument("--test_path", default="./dataset/test_hyper.tsv", type=str, help="The test dataset path.")
parser.add_argument("--save_dir", default="./checkpoint", type=str,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=20, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float,
                    help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

platform = 'Paddle'
executor = 'HypernymDiscovery'
model = 'HypernymDiscovery'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
hypernym_discovery = executor(args=args)
hypernym_discovery.run()

print("-----------------------------------------------")
