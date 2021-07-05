import torch
import ujson as json
import numpy as np
import sys
import random
from tqdm import tqdm
from collections import Counter
from .util import get_batch, get_patterns, merge_batch, weight_init

from .models.pat_match import Pat_Match
from .models.soft_match_bert import SoftMatch

tqdm.monitor_interval = 0
np.set_printoptions(threshold=np.inf)


def read(config, unlabeled_data=None, test_data=None):

    def _read(path, dataset):
        res = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    d = json.loads(line)
                    res.append(d)
        return res
    def _read_mmd(entries):
        res = []
        for line in entries:
            line = line.strip()
            if len(line) > 0:
                d = json.loads(line)
                res.append(d)
        return res
    if unlabeled_data is None and test_data is None:
        train_data = _read(config.train_file, config.dataset)
        dev_data = _read(config.dev_file, config.dataset)
        test_data = _read(config.test_file, config.dataset)
    else:
        train_data = _read_mmd(unlabeled_data)
        dev_data = _read_mmd(test_data)
        test_data = _read_mmd(test_data)
    
    from .semeval_loader import read_glove, get_counter, token2id, read_data
    counter = get_counter(train_data)
    emb_dict = read_glove(config.word2vec_file, counter, config.glove_word_size, config.glove_dim)
    word2idx_dict, word_emb = token2id(config, counter, emb_dict)

    train_data = read_data(train_data)
    dev_data = read_data(dev_data)
    test_data = read_data(test_data)
    return word2idx_dict, word_emb, train_data, dev_data, test_data


def nero_run(config, data, match):
    word2idx_dict, word_emb, train_data, dev_data, test_data = data
    patterns = get_patterns(config, word2idx_dict)



    print('train patterns')

    from . import semeval_constant as constant
    regex = Pat_Match(config, constant.LABEL_TO_ID)
    # match = SoftMatch(config, word_mat=word_emb, word2idx_dict=word2idx_dict)
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() and int(config.gpu) >= 0 else "cpu")
    config.device = device

    match.to(device)
    # match.apply(weight_init)


    labeled_data = []
    unlabeled_data = []
    for x in train_data:
        batch = [x["tokens"]]
        res, pred = regex.match(batch)
        patterns["weights"] += res[0]
        if np.amax(res) > 0:
            x["rel"] = pred.tolist()[0]
            x["pat"] = np.argmax(res, axis=1).tolist()[0]
            labeled_data.append(x)
        else:
            x["rel"] = 0
            unlabeled_data.append(x)
    patterns["weights"] = patterns["weights"] / np.sum(patterns["weights"])
    """
    patterns：
    rels: [[0,1,0], ...], pattern_num * rels, 每条pattern对应的relation
    pats: embedding: 10*word_embedding_dim， 每条patterns的embedding
    weights: pattern的权重, 维度: patterns_num
    """
    random.shuffle(unlabeled_data)
    print("{} labeled data".format(len(labeled_data)))

    dev_history, test_history = [], []


    lr = float(config.init_lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, match.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=config.lr_decay)
    match.train()

    # import pdb;pdb.set_trace()

    for epoch in tqdm(range(1, config.num_epoch + 1), desc="Epoch"):
        idx = 0
        for batch1, batch2 in zip(get_batch(config, labeled_data, word2idx_dict), get_batch(config, unlabeled_data, word2idx_dict, pseudo=True)):
            batch = merge_batch(batch1, batch2)
            """
            8是batchsize
            sent: 8 * 110
            mid: 8 * 110
            rel: 8 * 3(3是rel_nums)
            pat: 8 (每个数字代表对应的pattern_id)
            """
            # import pdb; pdb.set_trace()
            golds, preds, val, loss = match(batch['raw'], patterns['raws'], batch['raw_mask'], patterns['raw_masks'],
                                             batch["sent"], batch["mid"], batch["rel"],
                                            batch["pat"], patterns["pattern_rels"], patterns["pats"], patterns["weights"])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(match.parameters(), config.grad_clip)
            optimizer.step()
            if idx % 2 == 0:
                print(loss.detach())
            idx += 1
            

        scheduler.step()
        if epoch % 5 == 0 and epoch > 0:
            (dev_acc, dev_rec, dev_f1), best_entro = log(config, dev_data, patterns, word2idx_dict, match, "dev")
            (test_acc, test_rec, test_f1), _ = log(
                config, test_data, patterns, word2idx_dict, match, "test", entropy=best_entro)

            print("acc: {}, rec: {}, f1: {}".format(dev_acc, dev_rec, dev_f1))
            print("acc: {}, rec: {}, f1: {}".format(test_acc, test_rec, test_f1))
            dev_history.append((dev_acc, dev_rec, dev_f1))
            test_history.append((test_acc, test_rec, test_f1))

        
        # if (len(dev_history) >= 1 and dev_f1 > dev_history[-1][2]) or len(dev_history) == 1:
        #     checkpoint_path = config.checkpoint
        #     saver.save(sess, checkpoint_path)

    max_idx = dev_history.index(max(dev_history, key=lambda x: x[2]))
    max_acc, max_rec, max_f1 = test_history[max_idx]
    print("acc: {}, rec: {}, f1: {}".format(max_acc, max_rec, max_f1))
    sys.stdout.flush()
    return max_acc, max_rec, max_f1


def log(config, data, patterns, word2idx_dict, model, label="train", entropy=None):
    with torch.no_grad():
        golds, preds, vals = [], [], []
        for batch in get_batch(config, data, word2idx_dict):
            gold, pred, val, loss = model(batch['raw'], patterns['raws'], batch['raw_mask'], patterns['raw_masks'],
                                            batch["sent"], batch["mid"], batch["rel"], batch["pat"], 
                                            patterns["pattern_rels"], patterns["pats"], patterns["weights"], is_train=False)
            golds += gold.tolist()
            preds += pred.tolist()
            vals += val.tolist()


    threshold = [0.1 * i for i in range(1, 200)]
    acc, recall, f1 = 0., 0., 0.
    best_entro = 0.

    if entropy is None:
        for t in threshold:
            _preds = (np.asarray(vals, dtype=np.float32) <= t).astype(np.int32) * np.asarray(preds, dtype=np.int32)
            _preds = _preds.tolist()
            _acc, _recall, _f1 = evaluate(golds, _preds)
            if _f1 > f1:
            # if _acc > acc:
                acc, recall, f1 = _acc, _recall, _f1
                best_entro = t
    else:
        preds = (np.asarray(vals, dtype=np.float32) <= entropy).astype(np.int32) * np.asarray(preds, dtype=np.int32)
        preds = preds.tolist()
        acc, recall, f1 = evaluate(golds, preds)
    return (acc, recall, f1), best_entro


def evaluate(key, prediction):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == 0 and guess == 0:
            pass
        elif gold == 0 and guess != 0:
            guessed_by_relation[guess] += 1
        elif gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        elif gold != 0 and guess != 0:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro





