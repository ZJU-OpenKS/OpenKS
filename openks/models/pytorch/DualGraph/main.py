import warnings
warnings.filterwarnings('ignore')

from arguments import arg_parse
import math
import random
import numpy as np
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
# from torch_geometric.utils import degree
# from torch_geometric.transforms import OneHotDegree
from model.aug import TUDataset_aug
from model.predictor import Predictor
from model.selector import Selector
from model.trainer import Trainer, evaluate
from selection import get_relation_distribution, select_samples
from utils import scorer, torch_utils


def seed_everything(seed, cuda):
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_best_model(model_dir, model_type="predictor"):
    model_file = model_dir + "/best_model.pt"
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    if model_type == "predictor":
        predictor = Predictor(model_opt)
        model = Trainer(model_opt, predictor, model_type=model_type)
    else:
        selector = Selector(model_opt)
        model = Trainer(model_opt, selector, model_type=model_type)
    model.load(model_file)
    return model


args = arg_parse()
opt = vars(args)
seed_everything(opt["seed"], opt["cuda"])

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', opt["DS"])

# ori_dataset
dataset = TUDataset(path, name=opt["DS"], cleaned=False)

opt["num_features"] = max(dataset.num_features, 1)
opt["num_classes"] = dataset.num_classes
label_distribution = get_relation_distribution(dataset)

# if dataset.data.x is None or dataset.data.x.shape[1] == 0: # torch.Size([num, 0])
#     max_degree = 0
#     for data in dataset:
#         deg = degree(data.edge_index[1], num_nodes=data.num_nodes)
#         max_degree = max(max_degree, max(deg).item())
#     dataset.transform = OneHotDegree(int(max_degree))
# dataset_ori = [data for data in dataset]

# max_num = 100
# data_tmp = []
# label_list = dataset.data.y.tolist()
# for l in range(0, opt["num_classes"]):
#     idx = 0
#     for i, value in enumerate(label_list):
#         if value == l:
#             data_tmp.append(dataset[i])
#             idx += 1
#             if idx >= max_num:
#                 break
# label_tmp = []
# for data in data_tmp:
#     label_tmp.append(data.y.item())
# import collections
# counter = collections.Counter(label_tmp)

if dataset.data.x is None or dataset.data.x.shape[1] == 0: # torch.Size([num, 0])
    tmp = []
    for i in range(len(dataset)):
        x = torch.ones((dataset[i].num_nodes, 1))
        if dataset[i].edge_attr == None:
            tmp.append(Data(x=x, y=dataset[i].y, edge_index=dataset[i].edge_index))
        else:
            tmp.append(Data(x=x, y=dataset[i].y, edge_index=dataset[i].edge_index, edge_attr=dataset[i].edge_attr))
    dataset = tmp
else:
    dataset = [data for data in dataset]

dataset_ori = dataset

# aug_dataset
dataset = TUDataset_aug(path, name=opt["DS"], cleaned=False, aug=opt["aug"])

if dataset.data.x is None or dataset.data.x.shape[1] == 0: # torch.Size([num, 0])
    tmp = []
    for i in range(len(dataset)):
        x = torch.ones((dataset[i].num_nodes, 1))
        if dataset[i].edge_attr == None:
            tmp.append(Data(x=x, y=dataset[i].y, edge_index=dataset[i].edge_index))
        else:
            tmp.append(Data(x=x, y=dataset[i].y, edge_index=dataset[i].edge_index, edge_attr=dataset[i].edge_attr))
    dataset = tmp
else:
    dataset = [data for data in dataset]


np.random.seed(seed=21)
idx = list(np.random.permutation(len(dataset)))

# label:0.1, unlabel:0.5, val:0.1, test:0.2
label_end = math.ceil(len(dataset)*0.1)
unlabel_start = math.ceil(len(dataset)*0.2)
unlabel_ratio_end = math.ceil(len(dataset)*0.7)
unlabel_end = math.ceil(len(dataset)*0.7)
val_end = math.ceil(len(dataset)*0.8)

train_label_idx = idx[:label_end]
train_unlabel_idx = idx[unlabel_start:unlabel_ratio_end]
valset_idx = idx[unlabel_end:val_end]
testset_idx = idx[val_end:]

train_labelset = [dataset[i] for i in train_label_idx]
train_unlabelset = [dataset[i] for i in train_unlabel_idx]
valset = [dataset_ori[i] for i in valset_idx]
testset = [dataset_ori[i] for i in testset_idx]

print("=" * 100)
print("num_epoch: {}".format(opt["num_epoch"]))
print("lr: {}".format(opt["lr"]))
print("hidden_dim: {}".format(opt["hidden_dim"]))
print("batch_size: {}".format(opt["batch_size"]))
print("num_features: {}".format(opt["num_features"]))
print("num_classes: {}".format(opt["num_classes"]))
print("Labeled #: {}, Unlabeled #: {}, Valid #: {}, Test #: {}".format(len(train_labelset), \
                                        len(train_unlabelset), len(valset), len(testset)))
print("=" * 100)

# ceil(x) returns the smallest integer greater than or equal to x
num_iters = math.ceil(1.0 / opt["sample_ratio"])
k_samples = math.ceil(len(train_unlabelset) * opt["sample_ratio"])
# k_samples = len(train_unlabelset)


val_acc_iter, test_acc_iter = [], []

for num_iter in range(num_iters + 1):
    print('')

    # ====================== #
    # Begin Train on Predictor
    # ====================== #
    print('=' * 100)
    print("Training on iteration #%d for DualGraph Predictor using %s ..." % (num_iter, opt["DS"]))
    opt["model_save_dir"] = opt["p_dir"]
    opt["dropout"] = opt["p_dropout"]
    print("")

    torch_utils.save_config(opt, opt["model_save_dir"] + "/config.json", verbose=True)

    # prediction module
    predictor = Predictor(opt)
    model = Trainer(opt, predictor, model_type="predictor")
    model.train(train_labelset, valset, train_unlabelset)

    # Evaluate
    best_model_p = load_best_model(opt["model_save_dir"], model_type="predictor")
    val_acc = evaluate(best_model_p, valset)[0]
    test_acc = evaluate(best_model_p, testset)[0]
    print('val_acc:' + str(val_acc))
    print('test_acc:' + str(test_acc))
    val_acc_iter.append(val_acc)
    test_acc_iter.append(test_acc)
    best_model_p = load_best_model(opt["p_dir"], model_type="predictor")

    if len(train_unlabelset) == 0:
        break

    # ====================== #
    # Begin Train on Selector
    # ====================== #
    best_model_s = None
    if opt["selector_model"] != "none":
        print('=' * 50)
        print("Training on iteration #%d for DualGraph Selector using %s ..." % (num_iter, opt["DS"]))
        opt["model_save_dir"] = opt["s_dir"]
        opt["dropout"] = opt["s_dropout"]
        print('')

        # save config
        torch_utils.save_config(opt, opt["model_save_dir"] + "/config.json", verbose=True)

        # selector model
        selector = Selector(opt)
        if opt["selector_model"] == "predictor":
            selector = Predictor(opt)
        model = Trainer(opt, selector, model_type=opt["selector_model"])
        model.train(train_labelset, valset, train_unlabelset)

        best_model_s = load_best_model(opt["s_dir"], model_type=opt["selector_model"])

    # ====================== #
    # Select New Instances
    # ====================== #
    new_examples, rest_examples = select_samples(opt, k_samples, label_distribution,
                                                 best_model_p, best_model_s, 
                                                 train_unlabelset)

    # update dataset
    train_labelset = train_labelset + new_examples
    train_unlabelset = rest_examples

scorer.print_table(val_acc_iter, test_acc_iter, header="Best val and test ACC for %s:" % (opt["DS"]))

result_file = open("./result_table_ICDE.txt", mode="a", encoding="utf-8")
result_file.write("%s"%opt['DS'] + "\n")
# result_file.write("%s"%opt['DS'] + "\t" + "%s"%opt['hidden_dim'] + "\t" + "%s"%opt['batch_size']+ "\n")
for i in range(len(val_acc_iter)):
    result_file.write(str(val_acc_iter[i])+"\t"+str(test_acc_iter[i])+"\n")
result_file.write("\n\n")
result_file.close()