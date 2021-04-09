import random
import numpy as np

def load_vocabulary(vocab):
    print("load vocab containing words: {}".format(len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w

def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = set()
    pre_bio = "O"
    v = ""
    for i, bio in enumerate(bio_seq):
        if (bio == "O"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = ""
        elif (bio[0] == "B"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = word_seq[i]
        elif (bio[0] == "I"):
            if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                if v != "": pairs.add((pre_bio[2:], v))
                v = ""
            else:
                v += word_seq[i]
        pre_bio = bio
    if v != "": pairs.add((pre_bio[2:], v))
    return pairs

def cal_f1_score(preds, golds):
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def cal_f1_score_org_pro(preds, golds):
    assert len(preds) == len(golds)
    p_sum_org = 0
    p_sum_pro = 0
    r_sum_org = 0
    r_sum_pro = 0
    hits_org = 0
    hits_pro = 0
    for pred, gold in zip(preds, golds):
        for p in pred:
            if p[0] == "企业":
                p_sum_org += 1
            elif p[0] == "产品":
                p_sum_pro += 1
        for g in gold:
            if g[0] == "企业":
                r_sum_org += 1
            elif g[0] == "产品":
                r_sum_pro += 1        
        for label in pred:
            if label in gold and label[0] == "企业":
                hits_org += 1
            if label in gold and label[0] == "产品":
                hits_pro += 1
    p_org = hits_org / p_sum_org if p_sum_org > 0 else 0
    r_org = hits_org / r_sum_org if r_sum_org > 0 else 0
    f1_org = 2 * p_org * r_org / (p_org + r_org) if (p_org + r_org) > 0 else 0
    p_pro = hits_pro / p_sum_pro if p_sum_pro > 0 else 0
    r_pro = hits_pro / r_sum_pro if r_sum_pro > 0 else 0
    f1_pro = 2 * p_pro * r_pro / (p_pro + r_pro) if (p_pro + r_pro) > 0 else 0
    return p_org, r_org, f1_org, p_pro, r_pro, f1_pro


class DataProcessor_LSTM(object):
    def __init__(self, 
                 input_data, 
                 output_data, 
                 w2i_char,
                 w2i_bio,
                 shuffling=False):
        
        inputs_seq = []
        for line in input_data:
            seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line]
            inputs_seq.append(seq)
        
        outputs_seq = []
        for line in output_data:
            seq = [w2i_bio[word] for word in line]
            outputs_seq.append(seq)
                    
        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))
        
        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)
        
    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
    
    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_batch = []
        
        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True
        
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))
        
        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(outputs_seq_batch, dtype="int32"))

class DataProcessor_LSTM_for_sentences(object):
    def __init__(self, 
                 sentences, 
                 w2i_char,
                 w2i_bio,
                 shuffling=False):
        
        inputs_seq = []
        outputs_seq = []

        for item in sentences:
            seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in item.split(" ")]
            label = [w2i_bio["O"] for word in item.split(" ")]
            inputs_seq.append(seq)
            outputs_seq.append(label)
                
        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))
        
        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)
        
    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
    
    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_batch = []
        
        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True
        
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))
        
        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(outputs_seq_batch, dtype="int32"))