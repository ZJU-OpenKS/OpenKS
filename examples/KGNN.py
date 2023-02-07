import warnings
warnings.filterwarnings('ignore')

from arguments import arg_parse
import math
import random
import numpy as np
import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

# from openks.models.pytorch.KGNN.grakel import GraphKernel
from openks.models.pytorch.KGNN.GraKeL.grakel.kernels.weisfeiler_lehman import WeisfeilerLehman
from openks.models.pytorch.KGNN.GraKeL.grakel.kernels.shortest_path import ShortestPath
from openks.models.pytorch.KGNN.GraKeL.grakel.kernels.graphlet_sampling import GraphletSampling

from openks.models.pytorch.KGNN.model.gnn import GNN
from openks.models.pytorch.KGNN.model.memnn import MemNN
from openks.models.pytorch.KGNN.model.trainer import Trainer, evaluate
from openks.models.pytorch.KGNN.selection import get_relation_distribution, select_samples
from openks.models.pytorch.KGNN.utils import scorer, torch_utils


def load_best_model(model_dir, model_type="gnn"):
    model_file = model_dir + "/best_model.pt"
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    if model_type == "gnn":
        gnn = GNN(model_opt)
        model = Trainer(model_opt, gnn, model_type=model_type)
    else:
        gk = MemNN(model_opt)
        model = Trainer(model_opt, gk, model_type=model_type)
    model.load(model_file)
    return model


args = arg_parse()
opt = vars(args)
# random.seed(opt["seed"])
# torch.manual_seed(opt["seed"])
# if opt["cuda"]:
#     torch.cuda.manual_seed(opt["seed"])

path = osp.join(osp.dirname(osp.realpath(__file__)), "data", opt["DS"])
dataset = TUDataset(path, name=opt["DS"], cleaned=False) # .shuffle()
opt["num_features"] = max(dataset.num_features, 1)
opt["num_class"] = dataset.num_classes
label_distribution = get_relation_distribution(dataset)

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
dataset = [dataset[i] for i in idx]

# label:0.2, unlabel:0.5, val:0.1, test:0.2
label_end = math.ceil(len(dataset)*0.1)
unlabel_start = math.ceil(len(dataset)*0.2)
unlabel_ratio_end = math.ceil(len(dataset)*0.7)
unlabel_end = math.ceil(len(dataset)*0.7)
val_end = math.ceil(len(dataset)*0.8)

train_labelset = dataset[:label_end]
# train_unlabelset = dataset[unlabel_start:unlabel_end]
train_unlabelset = dataset[unlabel_start:unlabel_ratio_end]
valset = dataset[unlabel_end:val_end]
testset = dataset[val_end:]

print("=" * 100)
print("base_kernel: {}".format(opt["base_kernel"]))
print("gnn_num_epoch: {}".format(opt["gnn_num_epoch"]))
print("memnn_num_epoch: {}".format(opt["memnn_num_epoch"]))
print("lr: {}".format(opt["lr"]))
print("hidden_dim: {}".format(opt["hidden_dim"]))
print("batch_size: {}".format(opt["batch_size"]))
# print("nystroem_dim: {}".format(opt["nystroem_dim"]))
print("num_features: {}".format(opt["num_features"]))
print("Labeled #: {}, Unlabeled #: {}, Valid #: {}, Test #: {}".format(len(train_labelset), \
                                        len(train_unlabelset), len(valset), len(testset)))
print("=" * 100)

# ceil(x) returns the smallest integer greater than or equal to x
num_iters = math.ceil(1.0 / opt["sample_ratio"])
k_samples = math.ceil(len(train_unlabelset) * opt["sample_ratio"])
# k_samples = len(train_unlabelset)

# data preparation for graph kernel
GK_x = torch_utils.load_data(opt["DS"], "data")
GK_y = [data.y.item() for data in dataset]
GK_x, GK_y = [GK_x[i] for i in idx], [GK_y[i] for i in idx]

# GK_train_label, GK_train_label_y = GK_x[:label_end], GK_y[:label_end]
# GK_train_unlabel, GK_train_unlabel_y = GK_x[unlabel_start:unlabel_ratio_end], GK_y[unlabel_start:unlabel_ratio_end]
# # GK_train_unlabel, GK_train_unlabel_y = GK[unlabel_start:unlabel_end], GK_y[unlabel_start:unlabel_end]
# GK_val, GK_val_y = GK_x[unlabel_end:val_end], GK_y[unlabel_end:val_end]
# GK_test, GK_test_y = GK_x[val_end:], GK_y[val_end:]

if opt["base_kernel"] == "subtree_wl":
    WL = WeisfeilerLehman()
    _, X_attribute = WL.fit_transform(GK_x, GK_y)
    if type(X_attribute[opt["wl_iter"]].X) is np.ndarray:
        X_attribute_list = X_attribute[opt["wl_iter"]].X.tolist()
    else:
        X_attribute_list = X_attribute[opt["wl_iter"]].X.todense().tolist()
elif opt["base_kernel"] == "shortest_path":
    SP = ShortestPath()
    _, X_attribute = SP.fit_transform(GK_x, GK_y)  
    X_attribute_list = X_attribute.tolist() 
elif opt["base_kernel"] == "graphlet":
    GS = GraphletSampling()
    _, X_attribute = GS.fit_transform(GK_x)  
    X_attribute_list = X_attribute.tolist()  

GK_train_label = X_attribute_list[:label_end]
GK_train_unlabel = X_attribute_list[unlabel_start:unlabel_ratio_end]
GK_val = X_attribute_list[unlabel_end:val_end]
GK_test = X_attribute_list[val_end:]
opt["node_attribute_dim"] = len(GK_train_label[0])

GK_train_label_y = GK_y[:label_end]
GK_train_unlabel_y = GK_y[unlabel_start:unlabel_ratio_end]
# GK_train_unlabel_y = GK_y[unlabel_start:unlabel_end]
GK_val_y = GK_y[unlabel_end:val_end]
GK_test_y = GK_y[val_end:]

val_acc_iter, test_acc_iter = [], []
for num_iter in range(num_iters+1):
    print("")

    # ====================== #
    # Begin Train on GNN
    # ====================== #
    print("=" * 100)
    print("Training on iteration # %d for KGNN GNN using %s ..." % (num_iter, opt["DS"]))
    opt["model_save_dir"] = opt["p_dir"]
    opt["dropout"] = opt["p_dropout"]
    print("")

    # save config
    torch_utils.save_config(opt, opt["model_save_dir"] + "/config.json", verbose=False)

    # GNN module
    gnn = GNN(opt)
    model = Trainer(opt, gnn, model_type="gnn")
    model.train(train_labelset, valset)

    # Evaluate
    best_model_p = load_best_model(opt["model_save_dir"], model_type="gnn")

    val_acc = evaluate(best_model_p, valset, evaluate_type="gnn")[0]
    test_acc = evaluate(best_model_p, testset, evaluate_type="gnn")[0]
    print("Iter %d val_acc: " % num_iter + str(val_acc))
    print("Iter %d test_acc: " % num_iter + str(test_acc))
    val_acc_iter.append(val_acc)
    test_acc_iter.append(test_acc)
    

    if len(train_unlabelset) == 0:
        break

    # ====================== #
    # Begin Train on GK
    # ====================== #
    print("=" * 50)
    print("Training on iteration # %d for KGNN GK using %s ..." % (num_iter, opt["DS"]))
    opt["model_save_dir"] = opt["q_dir"]
    opt["dropout"] = opt["q_dropout"]
    # if opt["use_Nystroem"] == False:
    #     opt["gk_features"] = len(GK_train_label)
    print("")

    # save config
    torch_utils.save_config(opt, opt["model_save_dir"] + "/config.json", verbose=False)

    # GK model
    gk = MemNN(opt)
    # if opt["use_Nystroem"] == False:
    #     gk_model = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3}, opt["base_kernel"]], 
    #                             normalize=True, Nystroem=False)
    # else:
    #     gk_model = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3}, opt["base_kernel"]], 
    #                             normalize=True, Nystroem=opt["nystroem_dim"])
    # K_train = gk_model.fit_transform(GK_train_label).tolist()
    # K_unlabel = gk_model.transform(GK_train_unlabel).tolist()
    # K_val = gk_model.transform(GK_val).tolist()
    # K_test = gk_model.transform(GK_test).tolist()

    model = Trainer(opt, gk, model_type="gk")
    
    train_label_data = [(GK_train_label, GK_train_label[i], GK_train_label_y[i]) for i in range(len(GK_train_label))]
    train_unlabel_data = [(GK_train_label, GK_train_unlabel[i], GK_train_unlabel_y[i]) for i in range(len(GK_train_unlabel))]
    val_data = [(GK_train_label, GK_val[i], GK_val_y[i]) for i in range(len(GK_val))]
    test_data = [(GK_train_label, GK_test[i], GK_test_y[i]) for i in range(len(GK_test))]
    model.train(train_label_data, val_data)

    best_model_q = load_best_model(opt["model_save_dir"], model_type="gk")

    # val_acc = evaluate(best_model_q, val_data, evaluate_type="gk")[0]
    # test_acc = evaluate(best_model_q, test_data, evaluate_type="gk")[0]
    # print("Iter %d val_acc: " % num_iter + str(val_acc))
    # print("Iter %d test_acc: " % num_iter + str(test_acc))
    # val_acc_iter.append(val_acc)
    # test_acc_iter.append(test_acc+float("1"))

    # ====================== #
    # Select New Instances
    # ====================== #
    new_examples, rest_examples, GK_new_examples, GK_new_examples_y, GK_rest_examples, \
    GK_rest_examples_y = select_samples(opt, k_samples, label_distribution, 
                                        model_p=best_model_p, model_q=best_model_q,
                                        train_unlabelset=train_unlabelset,
                                        train_unlabel_data=train_unlabel_data,
                                        GK_train_unlabel=GK_train_unlabel,
                                        GK_train_unlabel_y=GK_train_unlabel_y)

    # update dataset
    train_labelset = train_labelset + new_examples
    train_unlabelset = rest_examples
    GK_train_label = GK_train_label + GK_new_examples
    GK_train_label_y = GK_train_label_y + GK_new_examples_y
    GK_train_unlabel = GK_rest_examples
    GK_train_unlabel_y = GK_rest_examples_y

scorer.print_table(val_acc_iter, test_acc_iter, header="Best val and test ACC for %s:" % (opt["DS"]))

result_file = open("./result_table_wsdm.txt", mode="a", encoding="utf-8")
result_file.write("%s"%opt['DS'] + "\n")
# result_file.write("%s"%opt['DS'] + "\t" + "%s"%opt['base_kernel'] + "\n")
for i in range(len(val_acc_iter)):
    result_file.write(str(val_acc_iter[i])+"\t"+str(test_acc_iter[i])+"\n")
result_file.write("\n\n")
result_file.close()
