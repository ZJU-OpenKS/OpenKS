import random
import numpy as np
import json
import math
import torch.nn.init as init
import torch.nn as nn
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('/home/ps/disk_sdb/yyr/codes/NEROtorch/pretrain_models/bert', do_lower_case=True)


def get_pos(start, end, length, pad=None):
    res = list(range(-start, 0)) + [0] * (end - start + 1) + list(range(length - end - 1))
    if pad is not None:
        res += [0 for _ in range(pad - len(res))]
    return res


def get_mask(start, end, length, pad=None):
    res = [0] * start + [1] * (end - start + 1) + [0] * (length - end - 1)
    if pad is not None:
        res += [0 for _ in range(pad - len(res))]
    return res


def get_word(tokens, word2idx_dict, pad=None):
    res = []
    for token in tokens:
        i = 1
        for each in (token, token.lower(), token.capitalize(), token.upper()):
            if each in word2idx_dict:
                i = word2idx_dict[each]
        res.append(i)
    if pad is not None:
        res = res[:pad]
        res += [0 for _ in range(pad - len(res))]
    return res


def get_id(tokens, idx_dict, pad=None):
    if not isinstance(tokens, list):
        tokens = [tokens]
    res = [idx_dict[token] if token in idx_dict else 1 for token in tokens]
    if pad is not None:
        res += [0 for _ in range(pad - len(res))]
    return res


def get_patterns(config, word2idx_dict, filt=None):

    from . import semeval_constant as constant
    patterns = config.patterns
    rels, pats, raws, raw_masks = [], [], [], []
    for pattern in patterns:
        rel, pat = pattern
        rel_id = constant.LABEL_TO_ID[rel]
        if filt is not None:
            if rel_id in filt:
                continue
        rel = [0. for _ in range(config.num_class)]
        rel[rel_id] = 1.

        pat = pat.split()
        raw = ''.join(pat)
        raw = np.asarray(tokenizer.encode(raw, add_special_tokens=True, padding='max_length', max_length=128, truncation=True))
        raws.append(raw)
        raw_masks.append((raw>0))
        pat = get_word(pat, word2idx_dict, pad=10)
        rels.append(rel)
        pats.append(pat)
    num_pats = len(rels)
    rels = np.asarray(rels, dtype=np.float32)
    pats = np.asarray(pats, dtype=np.int32)
    weights = np.ones([num_pats], dtype=np.float32)
    raws = np.asarray(raws)
    raw_masks = np.array(raw_masks)
    """
    rel: [0,1,0]
    pat: embedding: 10*word_idx
    """
    return {"pattern_rels": rels, "pats": pats, "weights": weights, "raws": raws, "raw_masks": raw_masks}


def get_feeddict(model, batch, patterns, is_train=True):
    return {model.sent: batch["sent"], model.rel: batch["rel"], model.mid: batch["mid"],
            model.hard: batch["pat"], model.rels: patterns["rels"], model.pats: patterns["pats"],
            model.weight: patterns["weights"], model.is_train: is_train}

    """
    
    8是batchsize
    sent: 8 * 110
    mid: 8 * 110
    rel: 8 * 3(3是rel_nums)
    pat: 8, (每个数字代表对应的pattern_id), 如果无对应的，就是-1
                

    patterns：
    rels: [[0,1,0], ...], pattern_num * rels, 每条pattern对应的relation
    pats: pattern_num*token_len 每条patterns的token表示
    weights: pattern的权重, 维度: patterns_num
    """


def get_batch(config, data, word2idx_dict, rel_dict=None, shuffle=True, pseudo=False):
    if shuffle:
        random.shuffle(data)
    batch_size = config.pseudo_size if pseudo else config.gt_batch_size
    length = config.length
    batches = math.ceil(len(data) / batch_size)
    for i in range(batches):
        batch = data[i * batch_size: (i + 1) * batch_size]
        raw = np.asarray(list(map(lambda x: tokenizer.encode(''.join(x["tokens"]), add_special_tokens=True, padding='max_length', max_length=128, truncation=True), batch)))
        raw_mask = (raw>0)
        sent = np.asarray(list(map(lambda x: get_word(x["tokens"], word2idx_dict, pad=length), batch)), dtype=np.int32)
        mid = np.asarray(list(map(lambda x: get_word(x["tokens"][x["start"] - 1: x["end"] + 2], word2idx_dict, pad=length), batch)), dtype=np.int32)
        rel = np.asarray(list(map(lambda x: [1.0 if i == x["rel"] else 0. for i in range(config.num_class)], batch)), dtype=np.float32)
        pat = np.asarray(list(map(lambda x: x["pat"], batch)), dtype=np.int32)
        # yield {"sent": sent, "mid": mid, "rel": rel, "raw": raw, "pat": pat}
        """sent: batch * 110*1
            mid: batch * 110*1
            rel: batch * num_relations * 1
            pat: batch * 1: 就是对应哪条pattern， -1代表没有对应的
        """
        yield {"sent": sent, "mid": mid, "rel": rel, "pat": pat, "raw": raw, "raw_mask": raw_mask}


def merge_batch(batch1, batch2):
    batch = {}
    for key in batch1.keys():
        try:
            val1 = batch1[key]
            val2 = batch2[key]
            val = np.concatenate([val1, val2], axis=0)
            batch[key] = val
        except Exception as e:
            print(key)
            print(e)
    return batch


def sample_data(config, data):
    random.shuffle(data)
    num = len(data)
    return data[:int(config.sample * num)]



def weight_init(m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)

            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)