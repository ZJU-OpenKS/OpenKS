import numpy as np
from enum import Enum
import json
import random
import pickle

class MaskHierarchicalType(Enum):
    DOCUMENT = 0
    PARAGRAPH = 1
    SENTENCE = 2
    NGRAM = 3
    WORD = 4

with open("raw_data/heterographine/graph_text.txt", "r") as f:
    lines = f.read().splitlines()
with open("raw_data/heterographine/graph_text_test.txt", "r") as f:
    lines2 = f.read().splitlines()
with open("raw_data/heterographine/edge_text.json", "r") as f:
    edge_text = json.load(f)
data = []
count = 0
for l in lines:
    n1, r, n2 = l.split("\t")    
    if r == "develops_from":
        if n1 + "\t" + n2 in lines2:
            continue    
    t1, name1 = n1.split(" : ")
    t2, name2 = n2.split(" : ")
    doc = name1 + " " + edge_text[r] + " " + name2
    offset1 = 0
    len1 = len(name1)
    offset2 = len(name1 + " " + edge_text[r])
    len2 = len(name2) + 1
    doc_mask = [[(MaskHierarchicalType.WORD, offset1, len1)], [(MaskHierarchicalType.WORD, offset2, len2)]]
    data.append((doc, doc_mask))
    doc = name1 + " relates to " + name2
    offset1 = 0
    len1 = len(name1)
    offset2 = len(name1 + " relates to")
    len2 = len(name2) + 1
    doc_mask = [[(MaskHierarchicalType.WORD, offset1, len1)], [(MaskHierarchicalType.WORD, offset2, len2)]]
    data.append((doc, doc_mask))
random.shuffle(data)
train_data = data[:int(0.8 * len(data))]
valid_data = data[int(0.8 * len(data)):]
with open("data/heterographine/train.pkl", 'wb') as f:
    pickle.dump(train_data, f)
with open("data/heterographine/valid.pkl", 'wb') as f:
    pickle.dump(valid_data, f)
    
with open("raw_data/nell/graph_text.txt", "r") as f:
    lines = f.read().splitlines()
with open("raw_data/nell/graph_text_test.txt", "r") as f:
    lines2 = f.read().splitlines()
with open("raw_data/nell/edge_text.json", "r") as f:
    edge_text = json.load(f)
data = []
for l in lines:
    n1, r, n2 = l.split("\t")
    if r == "concept:competeswith":
        if n1 + "\t" + n2 in lines2:
            continue
    t1, name1 = n1.split(" : ")
    t2, name2 = n2.split(" : ")
    doc = name1 + " " + edge_text[r] + " " + name2
    offset1 = 0
    len1 = len(name1)
    offset2 = len(name1 + " " + edge_text[r])
    len2 = len(name2) + 1
    doc_mask = [[(MaskHierarchicalType.WORD, offset1, len1)], [(MaskHierarchicalType.WORD, offset2, len2)]]
    data.append((doc, doc_mask))
    doc = name1 + " relates to " + name2
    offset1 = 0
    len1 = len(name1)
    offset2 = len(name1 + " relates to")
    len2 = len(name2) + 1
    doc_mask = [[(MaskHierarchicalType.WORD, offset1, len1)], [(MaskHierarchicalType.WORD, offset2, len2)]]
    data.append((doc, doc_mask))
random.shuffle(data)
train_data = data[:int(0.8 * len(data))]
valid_data = data[int(0.8 * len(data)):]
with open("data/nell/train.pkl", 'wb') as f:
    pickle.dump(train_data, f)
with open("data/nell/valid.pkl", 'wb') as f:
    pickle.dump(valid_data, f)
