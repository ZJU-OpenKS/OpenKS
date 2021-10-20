import argparse
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys
import io

import collections
from typing import Optional, List, Union, Dict
from dataclasses import dataclass

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddlenlp.utils.log import logger
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer, LinearDecayWithWarmup
from ..model import Relation_ExtractionModel

# utils

def find_entity(text_raw, id_, predictions, tok_to_orig_start_index,
                tok_to_orig_end_index):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                      tok_to_orig_end_index[i + j] + 1])
            entity_list.append(entity)
    return list(set(entity_list))


def decoding(example_batch,
             id2spo,
             logits_batch,
             seq_len_batch,
             tok_to_orig_start_index_batch,
             tok_to_orig_end_index_batch):
    """
    model output logits -> formatted spo (as in data set file)
    """
    formatted_outputs = []
    for (i, (example, logits, seq_len, tok_to_orig_start_index, tok_to_orig_end_index)) in \
            enumerate(zip(example_batch, logits_batch, seq_len_batch, tok_to_orig_start_index_batch, tok_to_orig_end_index_batch)):

        logits = logits[1:seq_len +
                        1]  # slice between [CLS] and [SEP] to get valid logits
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        tok_to_orig_start_index = tok_to_orig_start_index[1:seq_len + 1]
        tok_to_orig_end_index = tok_to_orig_end_index[1:seq_len + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        text_raw = example['text']
        complex_relation_label = [8, 10, 26, 32, 46]
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue  # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": {
                                '@value': id2spo['object_type'][id_]
                            },
                            'subject_type': id2spo['subject_type'][id_],
                            "object": {
                                '@value': object_
                            },
                            "subject": subject_
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {
                            '@value': id2spo['object_type'][id_].split('_')[0]
                        }
                        if id_ in [8, 10, 32, 46
                                   ] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split(
                                '_')[1]] = find_entity(text_raw, id_affi + 55,
                                                       predictions,
                                                       tok_to_orig_start_index,
                                                       tok_to_orig_end_index)[0]
                            object_type_dict[id2spo['object_type'][
                                id_affi].split('_')[1]] = id2spo['object_type'][
                                    id_affi].split('_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": object_type_dict,
                            "subject_type": id2spo['subject_type'][id_],
                            "object": object_dict,
                            "subject": subject_
                        })

        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        zipfile_path = file_path + '.zip'
        f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        f.write(file_path)

    return zipfile_path


def get_precision_recall_f1(golden_file, predict_file):
    r = os.popen(
        'python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'.
        format(golden_file, predict_file))
    result = r.read()
    r.close()
    precision = float(
        re.search("\"precision\", \"value\":.*?}", result).group(0).lstrip(
            "\"precision\", \"value\":").rstrip("}"))
    recall = float(
        re.search("\"recall\", \"value\":.*?}", result).group(0).lstrip(
            "\"recall\", \"value\":").rstrip("}"))
    f1 = float(
        re.search("\"f1-score\", \"value\":.*?}", result).group(0).lstrip(
            "\"f1-score\", \"value\":").rstrip("}"))

    return precision, recall, f1

# extract_chinese_and_punct
LHan = [
    [0x2E80, 0x2E99],  # Han # So  [26] CJK RADICAL REPEAT, CJK RADICAL RAP
    [0x2E9B, 0x2EF3
     ],  # Han # So  [89] CJK RADICAL CHOKE, CJK RADICAL C-SIMPLIFIED TURTLE
    [0x2F00, 0x2FD5],  # Han # So [214] KANGXI RADICAL ONE, KANGXI RADICAL FLUTE
    0x3005,  # Han # Lm       IDEOGRAPHIC ITERATION MARK
    0x3007,  # Han # Nl       IDEOGRAPHIC NUMBER ZERO
    [0x3021,
     0x3029],  # Han # Nl   [9] HANGZHOU NUMERAL ONE, HANGZHOU NUMERAL NINE
    [0x3038,
     0x303A],  # Han # Nl   [3] HANGZHOU NUMERAL TEN, HANGZHOU NUMERAL THIRTY
    0x303B,  # Han # Lm       VERTICAL IDEOGRAPHIC ITERATION MARK
    [
        0x3400, 0x4DB5
    ],  # Han # Lo [6582] CJK UNIFIED IDEOGRAPH-3400, CJK UNIFIED IDEOGRAPH-4DB5
    [
        0x4E00, 0x9FC3
    ],  # Han # Lo [20932] CJK UNIFIED IDEOGRAPH-4E00, CJK UNIFIED IDEOGRAPH-9FC3
    [
        0xF900, 0xFA2D
    ],  # Han # Lo [302] CJK COMPATIBILITY IDEOGRAPH-F900, CJK COMPATIBILITY IDEOGRAPH-FA2D
    [
        0xFA30, 0xFA6A
    ],  # Han # Lo  [59] CJK COMPATIBILITY IDEOGRAPH-FA30, CJK COMPATIBILITY IDEOGRAPH-FA6A
    [
        0xFA70, 0xFAD9
    ],  # Han # Lo [106] CJK COMPATIBILITY IDEOGRAPH-FA70, CJK COMPATIBILITY IDEOGRAPH-FAD9
    [
        0x20000, 0x2A6D6
    ],  # Han # Lo [42711] CJK UNIFIED IDEOGRAPH-20000, CJK UNIFIED IDEOGRAPH-2A6D6
    [0x2F800, 0x2FA1D]
]  # Han # Lo [542] CJK COMPATIBILITY IDEOGRAPH-2F800, CJK COMPATIBILITY IDEOGRAPH-2FA1D

CN_PUNCTS = [(0x3002, "。"), (0xFF1F, "？"), (0xFF01, "！"), (0xFF0C, "，"),
             (0x3001, "、"), (0xFF1B, "；"), (0xFF1A, "："), (0x300C, "「"),
             (0x300D, "」"), (0x300E, "『"), (0x300F, "』"), (0x2018, "‘"),
             (0x2019, "’"), (0x201C, "“"), (0x201D, "”"), (0xFF08, "（"),
             (0xFF09, "）"), (0x3014, "〔"), (0x3015, "〕"), (0x3010, "【"),
             (0x3011, "】"), (0x2014, "—"), (0x2026, "…"), (0x2013, "–"),
             (0xFF0E, "．"), (0x300A, "《"), (0x300B, "》"), (0x3008, "〈"),
             (0x3009, "〉"), (0x2015, "―"), (0xff0d, "－"), (0x0020, " ")]
#(0xFF5E, "～"),

EN_PUNCTS = [[0x0021, 0x002F], [0x003A, 0x0040], [0x005B, 0x0060],
             [0x007B, 0x007E]]


class ChineseAndPunctuationExtractor(object):
    def __init__(self):
        self.chinese_re = self.build_re()

    def is_chinese_or_punct(self, c):
        if self.chinese_re.match(c):
            return True
        else:
            return False

    def build_re(self):
        L = []
        for i in LHan:
            if isinstance(i, list):
                f, t = i
                try:
                    f = chr(f)
                    t = chr(t)
                    L.append('%s-%s' % (f, t))
                except:
                    pass  # A narrow python build, so can't use chars > 65535 without surrogate pairs!

            else:
                try:
                    L.append(chr(i))
                except:
                    pass
        for j, _ in CN_PUNCTS:
            try:
                L.append(chr(j))
            except:
                pass

        for k in EN_PUNCTS:
            f, t = k
            try:
                f = chr(f)
                t = chr(t)
                L.append('%s-%s' % (f, t))
            except:
                raise ValueError()
                pass  # A narrow python build, so can't use chars > 65535 without surrogate pairs!

        RE = '[%s]' % ''.join(L)
        # print('RE:', RE.encode('utf-8'))
        return re.compile(RE, re.UNICODE)


# data_loader
InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels"
])


def parse_label(spo_list, label_map, tokens, tokenizer):
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    seq_len = len(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list:
        for spo_object in spo['object'].keys():
            # assign relation label
            if spo['predicate'] in label_map.keys():
                # simple relation
                label_subject = label_map[spo['predicate']]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object']['@value'])
            else:
                # complex relation
                label_subject = label_map[spo['predicate'] + '_' + spo_object]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object'][spo_object])

            subject_tokens_len = len(subject_tokens)
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            forbidden_index = None
            if subject_tokens_len > object_tokens_len:
                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        labels[index][label_subject] = 1
                        for i in range(subject_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        if forbidden_index is None:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_tokens):
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

            else:
                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        labels[index][label_object] = 1
                        for i in range(object_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        if forbidden_index is None:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_tokens):
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels


def convert_example_to_feature(
        example,
        tokenizer: BertTokenizer,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        label_map,
        max_length: Optional[int]=512,
        pad_to_max_length: Optional[bool]=None):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer._tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    if spo_list is not None:
        labels = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]

    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
            max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
            max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels), )

#数据集
class DuIEDataset(paddle.io.Dataset):
    """
    Dataset of DuIE.
    """

    def __init__(
            self,
            input_ids: List[Union[List[int], np.ndarray]],
            seq_lens: List[Union[List[int], np.ndarray]],
            tok_to_orig_start_index: List[Union[List[int], np.ndarray]],
            tok_to_orig_end_index: List[Union[List[int], np.ndarray]],
            labels: List[Union[List[int], np.ndarray, List[str], List[Dict]]]):
        super(DuIEDataset, self).__init__()

        self.input_ids = input_ids
        self.seq_lens = seq_lens
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.labels = labels

    def __len__(self):
        if isinstance(self.input_ids, np.ndarray):
            return self.input_ids.shape[0]
        else:
            return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": np.array(self.input_ids[item]),
            "seq_lens": np.array(self.seq_lens[item]),
            "tok_to_orig_start_index":
            np.array(self.tok_to_orig_start_index[item]),
            "tok_to_orig_end_index": np.array(self.tok_to_orig_end_index[item]),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels": np.array(
                self.labels[item], dtype=np.float32),
        }

    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: BertTokenizer,
                  max_length: Optional[int]=512,
                  pad_to_max_length: Optional[bool]=None):
        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(
            os.path.dirname(file_path), "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = (
            [] for _ in range(5))
        dataset_scale = sum(1 for line in open(
            file_path, 'r', encoding="UTF-8"))
        logger.info("Preprocessing data, loaded from %s" % file_path)
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature = convert_example_to_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    label_map, max_length, pad_to_max_length)
                input_ids.append(input_feature.input_ids)
                seq_lens.append(input_feature.seq_len)
                tok_to_orig_start_index.append(
                    input_feature.tok_to_orig_start_index)
                tok_to_orig_end_index.append(
                    input_feature.tok_to_orig_end_index)
                labels.append(input_feature.labels)

        return cls(input_ids, seq_lens, tok_to_orig_start_index,
                   tok_to_orig_end_index, labels)


@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """

    def __call__(self, examples: List[Dict[str, Union[list, np.ndarray]]]):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])
        seq_lens = np.stack([x['seq_lens'] for x in examples])
        tok_to_orig_start_index = np.stack(
            [x['tok_to_orig_start_index'] for x in examples])
        tok_to_orig_end_index = np.stack(
            [x['tok_to_orig_end_index'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])

        return (batched_input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)

# relation_extraction task
class BCELossForDuIE(nn.Layer):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask.unsqueeze(-1)
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        loss = loss.mean()
        return loss


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

@Relation_ExtractionModel.register("Relation_Extraction", "Paddle")
class Relation_ExtractionPaddle(Relation_ExtractionModel):
    def __init__(self, args, name: str = 'Relation_ExtractionModel', ):
        super().__init__()
        self.name = name
        self.args = args

    @paddle.no_grad()
    def evaluate(self,model, criterion, data_loader, file_path, mode):
        """
        mode eval:
        eval on development set and compute P/R/F1, called between training.
        mode predict:
        eval on development / test set, then write predictions to \
            predict_test.json and predict_test.json.zip \
            under args.data_path dir for later submission or evaluation.
        """
        example_all = []
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                example_all.append(json.loads(line))
        id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
        with open(id2spo_path, 'r', encoding='utf8') as fp:
            id2spo = json.load(fp)

        model.eval()
        loss_all = 0
        eval_steps = 0
        formatted_outputs = []
        current_idx = 0
        for batch in tqdm(data_loader, total=len(data_loader)):
            eval_steps += 1
            input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
            logits = model(input_ids=input_ids)
            mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
            loss = criterion(logits, labels, mask)
            loss_all += loss.numpy().item()
            probs = F.sigmoid(logits)
            logits_batch = probs.numpy()
            seq_len_batch = seq_len.numpy()
            tok_to_orig_start_index_batch = tok_to_orig_start_index.numpy()
            tok_to_orig_end_index_batch = tok_to_orig_end_index.numpy()
            formatted_outputs.extend(decoding(example_all[current_idx: current_idx+len(logits)],
                                            id2spo,
                                            logits_batch,
                                            seq_len_batch,
                                            tok_to_orig_start_index_batch,
                                            tok_to_orig_end_index_batch))
            current_idx = current_idx+len(logits)
        loss_avg = loss_all / eval_steps
        print("eval loss: %f" % (loss_avg))

        if mode == "predict":
            predict_file_path = os.path.join(args.data_path, 'predictions.json')
        else:
            predict_file_path = os.path.join(args.data_path, 'predict_eval.json')

        predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                        predict_file_path)

        if mode == "eval":
            precision, recall, f1 = get_precision_recall_f1(file_path,
                                                            predict_zipfile_path)
            os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
            return precision, recall, f1
        elif mode != "predict":
            raise Exception("wrong mode for eval func")


    def run(self):
        args = self.args
        paddle.set_device(args.device)
        rank = paddle.distributed.get_rank()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

        # Reads label_map.
        # read B I O的id
        label_map_path = os.path.join(args.data_path, "predicate2id.json")
        if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
            sys.exit("{} dose not exists or is not a file.".format(label_map_path))
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp) # dict
        num_classes = (len(label_map.keys()) - 2) * 2 + 2  # 由于object和subject的区别 B标签*2+2

        # Loads pretrained model BERT
        model = BertForTokenClassification.from_pretrained(
            args.model_name_or_path, num_classes=num_classes)
        model = paddle.DataParallel(model)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        criterion = BCELossForDuIE()

        # Loads dataset.
        train_dataset = DuIEDataset.from_file(
            os.path.join(args.data_path, 'train_data.json'), tokenizer,
            args.max_seq_length, True)
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        collator = DataCollator()
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collator,
            return_list=True)
        
        eval_file_path = os.path.join(args.data_path, 'dev_data.json')
        test_dataset = DuIEDataset.from_file(eval_file_path, tokenizer,
                                            args.max_seq_length, True)
        test_batch_sampler = paddle.io.BatchSampler(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_batch_sampler,
            collate_fn=collator,
            return_list=True)

        # Defines learning rate strategy.
        steps_by_epoch = len(train_data_loader)
        num_training_steps = steps_by_epoch * args.num_train_epochs
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                            args.warmup_ratio)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        # Starts training.
        global_step = 0
        logging_steps = 50
        save_steps = 10000
        tic_train = time.time()
        for epoch in range(args.num_train_epochs):
            print("\n=====start training of %d epochs=====" % epoch)
            tic_epoch = time.time()
            model.train()
            for step, batch in enumerate(train_data_loader):
                input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
                logits = model(input_ids=input_ids)
                mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and(
                    (input_ids != 2))
                loss = criterion(logits, labels, mask)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                loss_item = loss.numpy().item()
                global_step += 1
                
                if global_step % logging_steps == 0 and rank == 0:
                    print(
                        "epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                        % (epoch, args.num_train_epochs, step, steps_by_epoch,
                        loss_item, logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()

                if global_step % save_steps == 0 and rank == 0:
                    print("\n=====start evaluating ckpt of %d steps=====" %
                        global_step)
                    precision, recall, f1 = self.evaluate(
                        model, criterion, test_data_loader, eval_file_path, "eval")
                    print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                        (100 * precision, 100 * recall, 100 * f1))
                    print("saving checkpoing model_%d.pdparams to %s " %
                        (global_step, args.output_dir))
                    paddle.save(model.state_dict(),
                                os.path.join(args.output_dir,
                                            "model_%d.pdparams" % global_step))
                    model.train()  # back to train mode

            tic_epoch = time.time() - tic_epoch
            print("epoch time footprint: %d hour %d min %d sec" %
                (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

        # Does final evaluation.
        if rank == 0:
            print("\n=====start evaluating last ckpt of %d steps=====" %
                global_step)
            precision, recall, f1 = self.evaluate(model, criterion, test_data_loader,
                                            eval_file_path, "eval")
            print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                (100 * precision, 100 * recall, 100 * f1))
            paddle.save(model.state_dict(),
                        os.path.join(args.output_dir,
                                    "model_%d.pdparams" % global_step))
            print("\n=====training complete=====")