import copy
import csv
import logging
import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DictDataset(Dataset):

    def __init__(self, **kwargs):
        self.data = kwargs
        self.data_len = None
        for v in kwargs.values():
            if self.data_len is None:
                self.data_len = v.size(0)
            else:
                assert self.data_len == v.size(0)

    def __getitem__(self, index):
        res = {}
        for (k, v) in self.data.items():
            res[k] = v[index]
        return res

    def __len__(self):
        return self.data_len


class AlternateDataset(Dataset):

    def __init__(self, *args):
        self.datasets = args
        self.num_alternatives = len(args)
        self.data_len = None
        for d in args:
            if self.data_len is None:
                self.data_len = len(d)
            else:
                assert self.data_len == len(d)
        self.counters = [0] * self.data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        res = self.datasets[self.counters[index] % self.num_alternatives][index]
        self.counters[index] += 1
        return res


class InputExample(object):

    def __init__(self, guid, text_a, text_b, text_c, head, rel, tail, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.head = head
        self.rel = rel
        self.tail = tail
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, label_id, pos_indicator=None, corrupted_part=-1):
        self.input_ids = input_ids
        self.label_id = label_id
        self.pos_indicator = pos_indicator
        self.corrupted_part = corrupted_part


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, return_label=False):
        labels = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 3:
                    labels.append(line[3])
                    line = line[:3]
                lines.append(line)
            if return_label:
                return (lines, labels)
            return lines


class KGProcessor(DataProcessor):

    def __init__(self, data_args, tokenizer, is_world_master):
        self.data_dir = data_args.data_dir
        self.data_split = data_args.data_split
        self.rank = data_args.rank
        self.num_split = data_args.num_split
        self.tokenizer = tokenizer
        self.is_world_master = is_world_master
        self.only_corrupt_entity = data_args.only_corrupt_entity
        self.vocab_size = len(tokenizer)
        self.max_seq_length = data_args.max_seq_length
        self.data_cache_dir = data_args.data_cache_dir if data_args.data_cache_dir is not None else data_args.data_dir
        os.makedirs(self.data_cache_dir, exist_ok=True)
        self.num_neg = data_args.num_neg

        self.build_ent()
        self.build_rel()
        self.ent_size = len(self.ent_list)
        self.rel_size = len(self.rel_list)
        self.name2id = {e: i + self.vocab_size for (i, e) in enumerate(self.ent_list)}
        self.id2name = {i + self.vocab_size: e for (i, e) in enumerate(self.ent_list)}
        self.name2id.update({r: i + self.vocab_size + self.ent_size for (i, r) in enumerate(self.rel_list)})
        self.id2name.update({i + self.vocab_size + self.ent_size: r for (i, r) in enumerate(self.rel_list)})
        assert len(self.name2id) == len(self.id2name) == self.ent_size + self.rel_size
        if data_args.type_constrain:
            self.build_type_constrain()

    def get_train_examples(self, epoch):
        data_dir = self.data_dir
        cached_example_path = os.path.join(self.data_cache_dir, f'cached_train_examples_neg{self.num_neg}_epoch{epoch}')
        os.makedirs(cached_example_path, exist_ok=True)
        (examples, features) = self._create_examples_and_features(
            os.path.join(data_dir, 'train.tsv'),
            cached_example_path,
            self.num_neg
        )
        return (examples, features)

    def get_dev_examples(self):
        data_dir = self.data_dir
        cached_example_path = os.path.join(self.data_cache_dir, f'cached_dev_examples_{self.num_neg}')
        os.makedirs(cached_example_path, exist_ok=True)
        (examples, features) = self._create_examples_and_features(
            os.path.join(data_dir, 'dev.tsv'),
            cached_example_path,
            self.num_neg
        )
        return (examples, features)

    def get_test_examples(self):
        data_dir = self.data_dir
        cached_example_path = os.path.join(self.data_cache_dir, 'cached_test_examples')
        os.makedirs(cached_example_path, exist_ok=True)
        if self.data_split:
            (examples, features) = self._create_examples_and_features(
                os.path.join(data_dir, f'test-p{self.rank + 1}-of-{self.num_split}.tsv'),
                cached_example_path
            )
        else:
            (examples, features) = self._create_examples_and_features(
                os.path.join(data_dir, 'test.tsv'),
                cached_example_path
            )
        return (examples, features)

    def build_ent(self):
        ent_cache_file = os.path.join(self.data_cache_dir, 'entity.pt')
        if os.path.exists(ent_cache_file):
            logger.info('loading entity data from {}'.format(ent_cache_file))
            (self.ent2text, self.ent2tokens) = torch.load(ent_cache_file)
        else:
            logger.info('building entity data')
            self.ent2text = {}
            self.ent2tokens = {}
            with open(os.path.join(self.data_dir, 'entity2text.txt'), 'r') as f:
                ent_lines = f.readlines()
                for line in tqdm(ent_lines, disable=not self.is_world_master):
                    tmp = line.strip().split('\t')
                    if len(tmp) == 2:
                        self.ent2text[tmp[0]] = tmp[1]
                        self.ent2tokens[tmp[0]] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tmp[1]))
            if self.data_dir.find('FB15') != -1:
                with open(os.path.join(self.data_dir, 'entity2textlong.txt'), 'r') as f:
                    ent_lines = f.readlines()
                    for line in tqdm(ent_lines, disable=not self.is_world_master):
                        tmp = line.strip().split('\t')
                        self.ent2text[tmp[0]] = tmp[1]
                        self.ent2tokens[tmp[0]] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tmp[1]))
            logger.info('saving entity data to {}'.format(ent_cache_file))
            if self.is_world_master:
                torch.save((self.ent2text, self.ent2tokens), ent_cache_file)
        self.ent_list = sorted(self.ent2text.keys())

    def build_rel(self):
        rel_cache_file = os.path.join(self.data_cache_dir, 'relation.pt')
        if os.path.exists(rel_cache_file):
            logger.info('loading relation data from {}'.format(rel_cache_file))
            (self.rel2text, self.rel2tokens) = torch.load(rel_cache_file)
        else:
            logger.info('building relation data')
            self.rel2text = {}
            self.rel2tokens = {}
            with open(os.path.join(self.data_dir, 'relation2text.txt'), 'r') as f:
                rel_lines = f.readlines()
                for line in tqdm(rel_lines, disable=not self.is_world_master):
                    temp = line.strip().split('\t')
                    self.rel2text[temp[0]] = temp[1]
                    self.rel2tokens[temp[0]] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(temp[1]))
            logger.info('saving relation data to {}'.format(rel_cache_file))
            if self.is_world_master:
                torch.save((self.rel2text, self.rel2tokens), rel_cache_file)
        self.rel_list = sorted(self.rel2text.keys())

    def build_type_constrain(self):
        KE_id2ent = {}
        with open(os.path.join(self.data_dir, 'entity2id.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            (emid, ent_id) = line.strip().split('\t')
            KE_id2ent[ent_id] = emid
        KE_id2rel = {}
        with open(os.path.join(self.data_dir, 'relation2id.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            (rmid, rel_id) = line.strip().split('\t')
            KE_id2rel[rel_id] = rmid
        with open(os.path.join(self.data_dir, 'type_constrain.txt'), 'r') as f:
            lines = f.readlines()
        (self.rel2valid_head, self.rel2valid_tail) = ({}, {})
        for (num_line, line) in enumerate(lines[1:]):
            line = line.strip().split('\t')
            relation = KE_id2rel[line[0]]
            ents = [KE_id2ent[ent] for ent in line[2:]]
            assert len(ents) == int(line[1])
            if num_line % 2 == 0:
                self.rel2valid_head[relation] = ents
            else:
                self.rel2valid_tail[relation] = ents

    def get_name2id(self):
        return self.name2id

    def get_id2name(self):
        return self.id2name

    def get_ent2text(self):
        return self.ent2text

    def get_rel2text(self):
        return self.rel2text

    def get_labels(self):
        return ['0', '1']

    def get_entities(self):
        return self.ent_list

    def get_relations(self):
        return self.rel_list

    def get_train_triples(self):
        return self._read_tsv(os.path.join(self.data_dir, 'train.tsv'))

    def get_dev_triples(self, return_label=False):
        return self._read_tsv(os.path.join(self.data_dir, 'dev.tsv'), return_label=return_label)

    def get_test_triples(self, return_label=False):
        if self.data_split:
            return self._read_tsv(
                os.path.join(self.data_dir, f'test-p{self.rank + 1}-of-{self.num_split}.tsv'),
                return_label=return_label
            )
        else:
            return self._read_tsv(os.path.join(self.data_dir, 'test.tsv'), return_label=return_label)

    def create_examples(self, lines, num_corr, print_info=True):
        if isinstance(lines, str):
            lines = self._read_tsv(lines)
        ent2text = self.ent2text
        entities = self.ent_list
        rel2text = self.rel2text
        relations = self.rel_list
        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(tqdm(lines, disable=not self.is_world_master or not print_info)):
            (head, rel, tail) = line
            head_ent_text = ent2text[head]
            tail_ent_text = ent2text[tail]
            relation_text = rel2text[rel]
            guid = [i, 0, 0]
            text_a = head_ent_text
            text_b = relation_text
            text_c = tail_ent_text
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, text_c=text_c,
                    head=head, rel=rel, tail=tail, label='0'
                )
            )
            if num_corr == 0:
                continue
            if self.only_corrupt_entity:
                assert num_corr == 1, 'should use only 1 negative sample when only corrupt entity'
                rnd = random.random()
            if not self.only_corrupt_entity or rnd <= 0.5:
                guid = [i, 1]
                for j in range(num_corr):
                    while True:
                        tmp_head = random.choice(self.ent_list)
                        tmp_triple_str = tmp_head + '\t' + rel + '\t' + tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_head_text = ent2text[tmp_head]
                    examples.append(
                        InputExample(
                            guid=guid + [j], text_a=tmp_head_text, text_b=text_b, text_c=text_c,
                            head=tmp_head, rel=rel, tail=tail, label='1'
                        )
                    )
            if not self.only_corrupt_entity:
                guid = [i, 2]
                for j in range(num_corr):
                    while True:
                        tmp_rel = random.choice(self.rel_list)
                        tmp_triple_str = head + '\t' + tmp_rel + '\t' + tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_rel_text = rel2text[tmp_rel]
                    examples.append(
                        InputExample(
                            guid=guid + [j], text_a=text_a, text_b=tmp_rel_text, text_c=text_c,
                            head=head, rel=tmp_rel, tail=tail, label='1'
                        )
                    )
            if not self.only_corrupt_entity or rnd > 0.5:
                guid = [i, 3]
                for j in range(num_corr):
                    while True:
                        tmp_tail = random.choice(self.ent_list)
                        tmp_triple_str = head + '\t' + rel + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    examples.append(
                        InputExample(
                            guid=guid + [j], text_a=text_a, text_b=text_b, text_c=tmp_tail_text,
                            head=head, rel=rel, tail=tmp_tail, label='1'
                        )
                    )
        return examples

    def _create_examples_and_features(self, lines, cache_path=None, num_corr=0):
        if cache_path is None:
            examples = self.create_examples(lines, num_corr, print_info=False)
            features = self.convert_examples_to_features(examples, print_info=False)
            return (examples, features)
        cache_example_file = os.path.join(cache_path, 'example.pt')
        cache_feature_file = os.path.join(cache_path, f'feature_{self.max_seq_length}.pt')
        if os.path.exists(cache_example_file):
            logger.info('loading examples from {}'.format(cache_example_file))
            examples = torch.load(cache_example_file)
            logger.info('load examples done')
        else:
            examples = self.create_examples(lines, num_corr)
            logger.info('saving examples to {}'.format(cache_example_file))
            if self.is_world_master:
                torch.save(examples, cache_example_file)
            logger.info('save examples done')
        if os.path.exists(cache_feature_file):
            logger.info('loading features from {}'.format(cache_feature_file))
            features = torch.load(cache_feature_file)
            logger.info('load features done')
        else:
            features = self.convert_examples_to_features(examples)
            logger.info('saving features to {}'.format(cache_feature_file))
            if self.is_world_master:
                torch.save(features, cache_feature_file)
            logger.info('save features done')
        return (examples, features)

    def tokenize(self, example):
        tokens_a = copy.deepcopy(self.ent2tokens[example.head])
        tokens_b = copy.deepcopy(self.rel2tokens[example.rel])
        tokens_c = copy.deepcopy(self.ent2tokens[example.tail])
        return (tokens_a, tokens_b, tokens_c)

    def convert_examples_to_features(self, examples, print_info=True):
        label_list = self.get_labels()
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        label_map = {label: i for (i, label) in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(tqdm(examples, disable=not self.is_world_master or not print_info)):
            corrupted_part = example.guid[1] - 1
            SEP_id = tokenizer.sep_token_id
            CLS_id = tokenizer.cls_token_id


            (tokens_a, tokens_b, tokens_c) = self.tokenize(example)
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 3)

            input_ids = [CLS_id] + tokens_a + tokens_b + [SEP_id] + tokens_c + [SEP_id]
            pos_indicator = (
                0,
                1 + len(tokens_a),
                len(tokens_a),
                1 + len(tokens_a) + len(tokens_b),
                1 + len(tokens_a) + len(tokens_b),
                2 + len(tokens_a) + len(tokens_b) + len(tokens_c)
            )

            label_id = label_map[example.label]
            input_ids += [0] * (max_seq_length - len(input_ids))
            if ex_index < 5 and print_info:
                logger.info('*** Example ***')
                logger.info('guid: %s' % example.guid)
                logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
                logger.info('label: %s (id = %d)' % (example.label, label_id))
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    label_id=label_id,
                    pos_indicator=pos_indicator,
                    corrupted_part=corrupted_part
                )
            )
        return features

    def get_dataset(self, args):
        (train_dataset, eval_dataset, predict_dataset) = (None, None, None)
        train_data_file = f'train_dataset_{self.num_neg}_{self.max_seq_length}.pt'
        dev_data_file = f'dev_dataset_{self.num_neg}_{self.max_seq_length}.pt'
        test_data_file = f'test_dataset_{self.num_neg}_{self.max_seq_length}.pt'
        train_data_file = os.path.join(self.data_cache_dir, train_data_file)
        dev_data_file = os.path.join(self.data_cache_dir, dev_data_file)
        test_data_file = os.path.join(self.data_cache_dir, test_data_file)
        if args.do_train:
            if os.path.exists(train_data_file):
                logger.info(f'loading train dataset from{train_data_file}')
                train_dataset = torch.load(train_data_file)
                logger.info('loading done')
            else:
                train_dataset = []
                for epoch in range(int(args.num_train_epochs) + 1):
                    logger.info(f'getting train features @ epoch {epoch}')
                    (train_examples, train_features) = self.get_train_examples(epoch)
                    logger.info(f'building train tensors @ epoch {epoch}')
                    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                    all_pos_indicator = torch.tensor([f.pos_indicator for f in train_features], dtype=torch.long)
                    all_corrupted_part = torch.tensor([f.corrupted_part for f in train_features], dtype=torch.long)
                    logger.info(f'buiding train dataset @ epoch {epoch}')
                    train_dataset.append(
                        DictDataset(
                            input_ids=all_input_ids,
                            labels=all_label_ids,
                            pos_indicator=all_pos_indicator,
                            corrupted_part=all_corrupted_part
                        )
                    )
                train_dataset = AlternateDataset(*train_dataset)
                if self.is_world_master:
                    torch.save(train_dataset, train_data_file)
                logger.info('build done')
        if args.do_eval:
            if os.path.exists(dev_data_file):
                eval_dataset = torch.load(dev_data_file)
            else:
                (eval_examples, eval_features) = self.get_dev_examples()
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_pos_indicator = torch.tensor([f.pos_indicator for f in eval_features], dtype=torch.long)
                eval_dataset = DictDataset(
                    input_ids=all_input_ids,
                    labels=all_label_ids,
                    pos_indicator=all_pos_indicator
                )
                if self.is_world_master:
                    torch.save(eval_dataset, dev_data_file)
        if args.do_predict:
            if os.path.exists(test_data_file):
                predict_dataset = torch.load(test_data_file)
            else:
                (eval_examples, eval_features) = self.get_test_examples()
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_pos_indicator = torch.tensor([f.pos_indicator for f in eval_features], dtype=torch.long)
                predict_dataset = DictDataset(
                    input_ids=all_input_ids,
                    labels=all_label_ids,
                    pos_indicator=all_pos_indicator
                )
                if self.is_world_master:
                    torch.save(predict_dataset, test_data_file)
        return (train_dataset, eval_dataset, predict_dataset)


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
    if total_length <= max_length:
        return False
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_c) >= len(tokens_a) and len(tokens_c) >= len(tokens_b):
            tokens_c.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
    return True
