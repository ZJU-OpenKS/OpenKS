# -*-coding:utf-8-*-

import os
import re
import json
import warnings
import random
from functools import partial
from tqdm import tqdm, trange

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from paddlenlp.metrics import AccuracyAndF1
# from utils import read_by_lines, write_by_lines, load_dict

import ast
import argparse

warnings.filterwarnings('ignore')


# yapf: enable.
class InputExample(object):
    def __init__(self, guid, e1, e2, sentence, label=None):
        '''
        a single training/test example for sequence pair classification
        :param guid:string, event pairs for feature sequence truncate processing
        :param e1:string, The untokenized text of the first event
        :param e2:string, The untokenized text of the second event
        :param (Optional)label:string. The label of the example.
        '''
        self.guid = guid
        self.e1 = e1
        self.e2 = e2
        self.sentence = sentence
        self.label = label


class PredInputExample(object):
    def __init__(self, guid, e1, e2, sentence, label=None):
        '''
        a single training/test example for sequence pair classification
        :param guid:string, event pairs for feature sequence truncate processing
        :param e1:string, The untokenized text of the first event
        :param e2:string, The untokenized text of the second event
        :param (Optional)label:string. The label of the example.
        '''
        self.guid = guid
        self.e1 = e1
        self.e2 = e2
        self.sentence = sentence


class SREsProcessor(object):
    '''
    processor for SeRI data set
    '''

    def get_train_examples(self, data_path):
        return self._create_examples(
            self.read_dat(data_path), 'train')

    def get_dev_examples(self, data_path):
        return self._create_examples(
            self.read_dat(data_path), 'dev')

    def get_test_examples(self, data_path):
        return self._create_examples(
            self.read_dat(data_path), 'test')

    def get_pred_examples(self, data_path):
        return self._create_pred_examples(
            self.read_dat(os.path.join(data_path)), 'test')

    def get_labels(self):
        '''
        SUB:1,SUP:-1,None:0
        :return:
        '''
        return {'SUB': 0, 'SUP': 2, 'None': 1}
        # return ['-1','0','1']

    def read_dat(self, input_file):
        '''
        样本中包含重复样本
        :param input_file:
        :return:
        '''
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    lines.append(line)
        li = list(set(lines))
        li.sort()
        return li

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            label, other = line.strip().split('\t', maxsplit=1)
            # -1~1变为1-2方便计算l
            label = int(label) + 1
            other = [i.strip() for i in other.strip().split('|||')]
            e1 = other[0]
            guid = "%s-%s" % (set_type, i)
            # same process with ge's feature
            other[2] = re.sub('[\\[\\]]+', ' ', other[2])  # filter [ ]
            other[2] = re.sub('(\\\'){3}', '', other[2])  # filter '''(\'\'\')
            other[2] = re.sub('\\|{1}', ' ', other[2])  # filter |
            other[1] = ' '.join(other[1].strip().split())  # 数据中包含多个空格分句错误
            other[2] = ' '.join(other[2].strip().split())
            # other[2]中包含多句，只提取包含e2_mention的一个子句
            sents_temp = re.split('(?i)(?<=[.?!])(?<![a-z]\.[a-z]\.)\\s+(?=[a-z])', other[2])
            exist_flag = False
            for sent in sents_temp:
                if other[1] in sent:
                    exist_flag = True
                    other[2] = sent
                    break
            if not exist_flag:
                logging.warning("No e2 exists in the description!")
                raise
            e2 = other[1]
            sentence = ' '.join([i for i in other[2:] if i])  # filter ''
            examples.append(
                InputExample(guid, e1=e1, e2=e2, sentence=sentence, label=label))
        return examples

    def _create_pred_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            other = line
            other = [i.strip() for i in other.strip().split('|||')]
            e1 = other[0]
            guid = "%s-%s" % (set_type, i)
            # same process with ge's feature
            other[2] = re.sub('[\\[\\]]+', ' ', other[2])  # filter [ ]
            other[2] = re.sub('(\\\'){3}', '', other[2])  # filter '''(\'\'\')
            other[2] = re.sub('\\|{1}', ' ', other[2])  # filter |
            other[1] = ' '.join(other[1].strip().split())  # 数据中包含多个空格分句错误
            other[2] = ' '.join(other[2].strip().split())
            # other[2]中包含多句，只提取包含e2_mention的一个子句
            sents_temp = re.split('(?i)(?<=[.?!])(?<![a-z]\.[a-z]\.)\\s+(?=[a-z])', other[2])
            exist_flag = False
            for sent in sents_temp:
                if other[1] in sent:
                    exist_flag = True
                    other[2] = sent
                    break
            if not exist_flag:
                logging.warning("No e2 exists in the description!")
                raise
            e2 = ' '.join([i for i in other[2:] if i])  # filter ''
            examples.append(
                PredInputExample(guid, e1=e1, e2=e2))
        return examples


@paddle.no_grad()
def evaluate(model, criterion, metric, num_label, data_loader):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        loss = paddle.mean(criterion(logits, labels))
        losses.append(loss.numpy())
        # preds = paddle.argmax(logits, axis=-1)
        correct = metric.compute(logits, labels)
        metric.update(correct)
        res = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return res, avg_loss


def convert_example_to_feature(example, tokenizer, label_vocab=None, max_seq_len=512, is_test=False):
    e1, e2, sentence, label = example
    e1_input = tokenizer(
        e1,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    e2_input = tokenizer(
        e2,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    sentence_input = tokenizer(
        sentence,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    e1_id = e1_input['input_ids'][1:-1]
    e2_id = e2_input['input_ids'][1:-1]
    sentence_id = sentence_input['input_ids'][1:-1]
    e1_e2_id = tokenizer.build_inputs_with_special_tokens(e1_id, e2_id)[1:-1]

    if (len(sentence_id) + len(e1_e2_id)) > max_seq_len - 3:
        if len(e1_e2_id) > (max_seq_len - 3) and len(sentence_id) > (max_seq_len - 3):
            e1_e2_id = e1_e2_id[0:max_seq_len / 3]
            sentence_id = sentence_id[0:max_seq_len - 3 - max_seq_len / 4]
        elif len(sentence_id) > len(e1_e2_id):
            sentence_id = sentence_id[0:max_seq_len - len(e1_e2_id) - 3]
        elif len(sentence_id) < len(e1_e2_id):
            e1_e2_id = e1_e2_id[0:max_seq_len - len(e1_e2_id) - 3]

    input_ids = tokenizer.build_inputs_with_special_tokens(e1_e2_id, sentence_id)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(e1_e2_id, sentence_id)
    seq_len = e1_input['seq_len'] + e2_input['seq_len'] + sentence_input['seq_len'] + 4

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        return input_ids, token_type_ids, seq_len, label


def predict_data_dealer(path):
    process = SREsProcessor
    pred_example = process.get_pred_examples(path)
    return pred_example


def pred2formot(pred, sent):
    temp_list = []
    temp = [None, -1, -1, None]
    for id, item in enumerate(pred):
        if item[0] != 'I' and temp[0] is not None:
            temp[3] = "".join(sent[int(temp[1]):int(temp[2])])
            temp_list.append(temp)
            temp = [None, -1, -1, None]
        if item[0] == 'B':
            temp[0] = item[2:]
            temp[1] = id
            temp[2] = id + 1
        elif item[0] == 'I' and item[2:] == temp[0]:
            temp[2] = id + 1
    return temp_list


class SubEventRecognization(paddle.io.Dataset):
    """DuEventExtraction"""

    def __init__(self, data_path, data_type):
        process = SREsProcessor()
        relation_dict = process.get_labels()
        self.label_vocab = relation_dict
        if data_type == 'train':
            examples = process.get_train_examples(data_path)
        elif data_type == 'test':
            examples = process.get_test_examples(data_path)

        self.e1_list = []
        self.e2_list = []
        self.sentence_list = []
        self.label_ids = []
        for example in examples:
            e1, e2, sentence, label = example.e1, example.e2, example.sentence, example.label,
            self.e1_list.append(e1.lower())
            self.e2_list.append(e2.lower())
            self.sentence_list.append(sentence.lower())
            self.label_ids.append(label)

        self.label_num = len(self.label_vocab)

    def __len__(self):
        return len(self.e1_list)

    def __getitem__(self, index):
        return self.e1_list[index], self.e2_list[index], self.sentence_list[index], self.label_ids[index]


def do_train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    os.makedirs(args.checkpoints, exist_ok=True)
    paddle.set_device(args.device)
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    train_dataset = SubEventRecognization(args.train_data_path, data_type='train')
    test_dataset = SubEventRecognization(args.test_data_path, data_type='test')

    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    causality_dict = SREsProcessor().get_labels()
    label_map = causality_dict

    model = ErnieForSequenceClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))
    model = paddle.DataParallel(model)

    print("============start train==========")

    trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        label_vocab=train_dataset.label_vocab,
        max_seq_len=args.max_seq_len,
        is_test=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),  # token type ids
        Stack(dtype='int32'),  # sequence lens
        Stack(dtype='int64')  # labels
    ): fn(list(map(trans_func, samples)))

    batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    # batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn)

    test_loader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=batchify_fn)

    num_training_steps = len(train_loader) * args.num_epoch

    print("_a_")
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    metric = AccuracyAndF1()
    criterion = paddle.nn.loss.CrossEntropyLoss()

    step, best_f1 = 0, 0.0
    model.train()

    for epoch in trange(args.num_epoch):
        print("__1__")
        for idx, (input_ids, token_type_ids, seq_lens, labels) in enumerate(train_loader):
            logits = model(input_ids, token_type_ids)
            loss = paddle.mean(criterion(logits, labels))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                metric_data, avg_loss = evaluate(model, criterion, metric, len(label_map), test_loader)
                acc, p, r, f1, _ = metric_data
                print(f'dev step: {step} - loss: {avg_loss:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                      f'f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    best_f1 = f1
                    print(f'==============================================save best model ' \
                          f'best performerence {best_f1:5f}')
                    paddle.save(model.state_dict(), '{}/best.pdparams'.format(args.checkpoints))
            step += 1

    # save the final model
    if rank == 0:
        paddle.save(model.state_dict(), '{}/final.pdparams'.format(args.checkpoints))


def do_predict(args):
    paddle.set_device(args.device)
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    process = SREsProcessor()
    relation_dict = process.get_labels()
    label_map = relation_dict
    pretrained_model_path = os.path.join(args.init_ckpt, "best.pdparams")

    model = ErnieForSequenceClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))
    id2label = {val: key for key, val in label_map.items()}

    print("============model loading==========")
    if not pretrained_model_path or not os.path.isfile(pretrained_model_path):
        raise Exception("init checkpoints {} not exist".format(pretrained_model_path))
    else:
        state_dict = paddle.load(pretrained_model_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % pretrained_model_path)

    # load data from predict file
    sentences = predict_data_dealer(args.predict_data)
    encoded_inputs_list = []
    for sent in sentences:
        e1, e2, sentence = sent.e1, sent.e2, sent.sentence
        input_ids, token_type_ids, seq_len = convert_example_to_feature([e1, e2, sent.sentence, []], tokenizer,
                                                                        max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),  # token_type_ids
        Stack(dtype='int64')  # sequence lens
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [encoded_inputs_list[i: i + args.batch_size]
                            for i in range(0, len(encoded_inputs_list), args.batch_size)]

    batch_sentence = [sentences[i: i + args.batch_size]
                      for i in range(0, len(encoded_inputs_list), args.batch_size)]
    results = []
    print("============start predict==========")
    model.eval()
    for batch, sentence_batch in zip(batch_encoded_inputs, batch_sentence):
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs_ids = paddle.argmax(logits, -1).numpy()
        probs = probs.numpy()
        for item, sentence in probs, sentence_batch:
            results.append(
                {"label": id2label(item), "e1:": sentence.e1, "e2:": sentence.e2, "sentence:": sentence.sentence})

    with open(args.predict_save_path, 'w') as pred_ouput_file:
        for item in results:
            pred_ouput_file.write(json.dumps(item) + '\n')


@Sequence_ExtractionModel.register("Sequence_Extraction","Paddle")
class SubEvent_ExtractionPaddle(Sequence_ExtractionModel):
    '''Base class for event extract trainer'''
    def __init__(self, args, name: str = 'Sequence_ExtractionModel'):
        self.name = name
        self.args = args

    def run(self):
        do_train(self.args)

    def pred(self):
        do_predict(self.args)