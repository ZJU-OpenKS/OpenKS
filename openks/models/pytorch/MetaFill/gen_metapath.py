import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
import ilm.tokenize_util
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from ilm.infer import infill_with_ilm
from classifier import GPTClassifier
import json

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def metapath_induction(args, paths, k):
    metapath_score = {}
    with open(args.edge_node, "r") as f:
        edge_node = json.load(f)
    for p in paths:
        metap = []
        flag = True
        for e in p:
            if len(e.split("->")) != 3:
                flag = False
                break
            n1, r, n2 = e.split("->")
            t1 = n1.split(" : ")[0]
            t2 = n2.split(" : ")[0]
            if r not in edge_node:
                flag = False
                break
            if [t1, t2] not in edge_node[r]:
                flag = False
                break
            metap.append(t1 + "-" + r + "-" + t2)
        if not flag:
            continue
        metapath = " ".join(metap)
        if metapath not in metapath_score:
            metapath_score[metapath] = 0
        metapath_score[metapath] += 1
    path_score_sorted = sorted(metapath_score.items(), key=lambda item:item[1], reverse=True)
    with open("meta_" + args.output_prefix + str(k + 1) + ".txt", "w") as f:
        for p, score in path_score_sorted:
            f.write(p + "\t" + str(score) + "\n")
            
def classify(node, pre, edge, classifier, tokenizer, id_label):
    h_text = pre.split(" : ")[1]
    t_text = node
    c_text = h_text + " [SEP] " + edge + " [SEP] " + t_text
    with torch.no_grad():
        try:
            test_input_h = tokenizer([h_text], padding="max_length", max_length = 200, truncation=True, return_tensors="pt")
            test_input_t = tokenizer([t_text], padding="max_length", max_length = 200, truncation=True, return_tensors="pt")
            test_context = tokenizer([c_text], padding="max_length", max_length = 200, truncation=True, return_tensors="pt")
            #logger.info(test_input_h)
            #logger.info(test_input_t)
            #logger.info(test_context)
            mask_h = test_input_h['attention_mask'].cuda()
            input_id_h = test_input_h['input_ids'].squeeze(1).cuda()
            mask_t = test_input_t['attention_mask'].cuda()
            input_id_t = test_input_t['input_ids'].squeeze(1).cuda()
            mask_context = test_context['attention_mask'].cuda()
            context = test_context['input_ids'].squeeze(1).cuda()     
            output_h, output_t = classifier(input_id_h, input_id_t, mask_h, mask_t, context, mask_context)   
        except IndexError:
            return "none"
        predict = output_t.argmax(dim=1).tolist()[0]
        pred = id_label[str(predict)]
    return pred

def run_model(args):
    pairs = []
    classifier = torch.load(args.classifier)
    classifier.eval()
    tokenizer_class = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer_class.pad_token is None:
        tokenizer_class.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer_class.sep_token is None:
        tokenizer_class.add_special_tokens({'sep_token': '[SEP]'})
    with open(args.node_in, "r") as f:
        node_in = json.load(f)
    with open(args.node_out, "r") as f:
        node_out = json.load(f)
    with open(args.type_out, "r") as f:
        type_out = json.load(f)
    with open(args.edge_text, "r") as f:
        edge_text = json.load(f)
    with open(args.id_label, "r") as f:
        id_label = json.load(f)
    with open(args.train_file, "r") as f:
        lines = f.read().splitlines()
    with open(args.graph_file, "r") as f:
        edges = set(f.read().splitlines())
    print(len(lines))
    for l in lines:
        pairs.append(l.split("\t"))
    print(len(pairs))
    tokenizer = ilm.tokenize_util.Tokenizer.GPT2
    with open(os.path.join(args.model_dir, 'additional_ids_to_tokens.pkl'), 'rb') as f:
        additional_ids_to_tokens = pickle.load(f)
    additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
    try:
        ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
    except ValueError:
        print('Already updated')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.eval()
    _ = model.to(device)
    for k in range(0, args.hop):
        paths = []
        if k == 0:
            for p in pairs:
                for j in range(args.n_sample):
                    start = p[0].split(" : ")[1]
                    end = p[1].split(" : ")[1]
                    if p[1] not in node_in:
                        break
                    end_edge_type = random.sample(node_in[p[1]], 1)[0]
                    if end_edge_type == args.edge_type:
                        continue
                    if p[0] + "\t" + end_edge_type + "\t" + p[1] in edges:
                        logger.info(start+ "->" + end_edge_type + "->" + end)
                        paths.append([p[0]+ "->" + end_edge_type + "->" + p[1]])
        else:
            for p in pairs:
                for j in range(args.n_sample):
                    #print(p[0])
                    start = p[0].split(" : ")[1]
                    end = p[1].split(" : ")[1]
                    #print(start,end)
                    path = []
                    context = start + " relates to" + " _. It relates to" * k + " " + end + ". "
                    if p[1] not in node_in:
                        break
                    end_edge_type = random.sample(node_in[p[1]], 1)[0]
                    edge = edge_text[end_edge_type]
                    # replace the last relates to
                    context = context[::-1].replace("relates to"[::-1], edge[::-1], 1)[::-1]
                    context_ids = ilm.tokenize_util.encode(context, tokenizer)
                    _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
                    pre = p[0]
                    flag = True
                    for i in range(k):
                        if i == 0:
                            if pre not in node_out:
                                flag = False
                                break
                            edge_type = random.sample(node_out[pre], 1)[0]
                        else:
                            pre_t = pre.split(" : ")[0]
                            if pre_t not in type_out:
                                flag = False
                                break
                            edge_type = random.sample(type_out[pre_t], 1)[0]
                        edge = edge_text[edge_type] 
                        context = context.replace("relates to", edge, 1)
                        position = context.find(" _")
                        pre_len = position
                        post_len = len(context) - position - 2
                        context_ids = ilm.tokenize_util.encode(context, tokenizer)
                        context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
                        logger.info(ilm.tokenize_util.decode(context_ids, tokenizer))
                        generated = infill_with_ilm(model, additional_tokens_to_ids, context_ids, num_infills=1)[0]
                        s = ilm.tokenize_util.decode(generated, tokenizer).replace("\n", " ")
                        logger.info(s)
                        node = s[pre_len: -post_len].strip()
                        if node == "":
                            flag = False
                            break
                        t = classify(node, pre, edge, classifier, tokenizer_class, id_label)
                        if t == "none":
                            flag = False
                            break
                        logger.info(t)
                        path.append(pre + "->" + edge_type + "->" + t + " : " + node)
                        pre = t + " : " + node
                        context = s
                    if flag:
                        path.append(pre + "->" + end_edge_type + "->" + p[1])
                        if len(path) == k + 1:
                            paths.append(path)
            
        with open(args.output_prefix + str(k + 1) + ".txt", "w") as f:
            for p in paths:
                f.write("\t".join(p) + "\n")            
        metapath_induction(args, paths, k)
                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default=None, help='txt file with node pairs in training set')
    parser.add_argument('--graph-file', type=str, default=None, help='txt file with all edges')
    parser.add_argument('--hop', type=int, default=5, help='number of hops from start to end')
    parser.add_argument('--model-dir', type=str, default='abs_ilm',
                        help='where the pretrained model stores')
    parser.add_argument('--output-prefix', type=str, default='path_',
                        help='output file name')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=10, help='sample times for each node pair')
    parser.add_argument('--node-in', type=str, default=None, help='legal edge types to each node')
    parser.add_argument('--node-out', type=str, default=None, help='legal edge types from each node')
    parser.add_argument('--type-out', type=str, default=None, help='legal edge types from each node type')
    parser.add_argument('--edge-text', type=str, default=None, help='legal edge type names')
    parser.add_argument('--edge-node', type=str, default=None, help='legal node types at the 2 end of each edge type')
    parser.add_argument('--classifier', type=str, default=None, help='context-aware node type classifier')
    parser.add_argument('--id-label', type=str, default=None, help='node types')
    parser.add_argument("--from-pretrained", action="store_true", help='whether to use our finetuned gpt')
    parser.add_argument('--edge-type', type=str, default=None, help='edge type')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.from_pretrained:
        # The class name was BertClassifier before
        from classifier import GPTClassifier as BertClassifier 
    run_model(args)
