import os
import sys
sys.path.append("./")
from openks.apps.irqa import irqa

def train_dual_encoder(train_conf_path):
    dual_encoder = irqa.load_model(model="zh_dureader_de", use_cuda=True, device_id=5, batch_size=32)
    dual_encoder.train(train_conf_path)

def train_cross_encoder(train_conf_path):
    cross_encoder = irqa.load_model(model="zh_dureader_ce", use_cuda=True, device_id=5, batch_size=32)
    cross_encoder.train(train_conf_path)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("USAGE: ")
        print ("      python3 irqa_train.py de|ce ${conf_path}")
        print ("for example:")
        print ("      python3 irqa_train.py de apps/irqa/conf/train_de_cn.conf")
        exit()

    model = sys.argv[1]
    conf_path = sys.argv[2]
    if model == 'de':
        train_dual_encoder(conf_path)
    elif model == 'ce':
        train_cross_encoder(conf_path)


