# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import os
import json
import warnings
import random
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from ..model import Event_ExtractionModel

# yapf: enable.

@paddle.no_grad()
def evaluate(model, criterion, metric, num_label, data_loader):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        loss = paddle.mean(criterion(logits.reshape([-1, num_label]), labels.reshape([-1])))
        losses.append(loss.numpy())
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss


def convert_example_to_feature(example, tokenizer, label_vocab=None, max_seq_len=512, no_entity_label="O", ignore_label=-1, is_test=False):
    tokens, labels, segments = example
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    #token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    segments = segments[:(max_seq_len - 2)]
    segments = [0] + segments + [0]

    if is_test:
        return input_ids, segments, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len-2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        return input_ids, segments, seq_len, encoded_label

def schema_loader(path):
    event_schema_list = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8').readlines()]
    event_type_dict = {'O': 0}
    role_type_dict = {'O': 0}
    trigger_label_num = 1
    argument_label_num = 1
    for event_schema in event_schema_list:
        if ('B-' + event_schema['event_type']) not in event_type_dict:
            event_type_dict['B-' + event_schema['event_type']] = trigger_label_num
            event_type_dict['I-' + event_schema['event_type']] = trigger_label_num + 1
            trigger_label_num += 2
        for role in event_schema['role_list']:
            if ('B-' + role['role']) not in role_type_dict:
                role_type_dict['B-' + role['role']] = argument_label_num
                role_type_dict['I-' + role['role']] = argument_label_num + 1
                argument_label_num += 2
    return event_type_dict, role_type_dict

def data_dealer(path, task_type):
    corpus = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8').readlines()]
    text_list = []
    label_ids_list = []
    segment_list = []
    for item in corpus:
        text = item['text']
        if task_type == 'trigger':
            label_ids = ['O'] * len(text)
            segment_ids = [0] * len(text)
            for event in item['event_list']:
                label_ids[event['trigger_start_index']] = 'B-' + event['event_type']
                for l in range(event['trigger_start_index'] + 1, event['trigger_start_index'] + len(event['trigger'])):
                    label_ids[l] = 'I-' + event['event_type']
            text_list.append(text)
            label_ids_list.append(label_ids)
            segment_list.append(segment_ids)
        elif task_type == 'argument':
            for event in item['event_list']:
                label_ids = ['O'] * len(text)
                segment_ids = [0] * len(text)
                for l in range(event['trigger_start_index'], event['trigger_start_index'] + len(event['trigger'])):
                    segment_ids[l] = 1
                for argument in event['arguments']:
                    label_ids[argument['argument_start_index']] = 'B-' + argument['role']
                    for l in range(argument['argument_start_index'] + 1,
                                   argument['argument_start_index'] + len(argument['argument'])):
                        label_ids[l] = 'I-' + argument['role']
                text_list.append(text)
                label_ids_list.append(label_ids)
                segment_list.append(segment_ids)
        else:
            assert False, "wrong task type"
    return text_list, label_ids_list, segment_list

def predict_data_dealer(path, data_type):
    corpus = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8').readlines()]
    text_list = []
    segment_list = []
    for item in corpus:
        text = item['text']
        if data_type == 'trigger':
            segment_ids = [0] * len(text)
            text_list.append(text)
            segment_list.append(segment_ids)
        elif data_type == 'argument':
            for event in item['event_list']:
                segment_ids = [0] * len(text)
                for l in range(event['trigger_start_index'], event['trigger_start_index'] + len(event['trigger'])):
                    segment_ids[l] = 1
                text_list.append(text)
                segment_list.append(segment_ids)
    return text_list, segment_list

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




class DuEventExtraction(paddle.io.Dataset):
    """DuEventExtraction"""
    def __init__(self, data_path, tag_path, task_type):
        trigger_dict, role_dict = schema_loader(tag_path)
        if task_type == 'trigger':
            self.label_vocab = trigger_dict
            text_list, label_ids_list, segment_list = data_dealer(data_path, task_type)
        elif task_type == 'argument':
            self.label_vocab = role_dict
            text_list, label_ids_list, segment_list = data_dealer(data_path, task_type)
        else:
            assert False, "wrong task type"
        self.word_ids = []
        self.label_ids = []
        for item in text_list:
            words = [t for t in list(item.lower())]
            self.word_ids.append(words)
        self.label_ids = label_ids_list
        self.segment_list = segment_list
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index], self.segment_list[index]


def do_train(args, task_type):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    os.makedirs(args.checkpoints, exist_ok=True)
    paddle.set_device(args.device)
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    no_entity_label = "O"
    ignore_label = -1

    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    trigger_dict, role_dict = schema_loader(args.tag_path)
    if task_type == 'trigger':
        label_map = trigger_dict
    elif task_type == 'argument':
        label_map = role_dict
    id2label = {val: key for key, val in label_map.items()}
    model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))
    model = paddle.DataParallel(model)

    print("============start train==========")
    train_ds = DuEventExtraction(args.train_data, args.tag_path, task_type)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path, task_type)
    #test_ds = DuEventExtraction(args.test_data, args.tag_path, task_type)

    trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        label_vocab=train_ds.label_vocab,
        max_seq_len=args.max_seq_len,
        no_entity_label=no_entity_label,
        ignore_label=ignore_label,
        is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'), # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'), # token type ids
        Stack(dtype='int64'), # sequence lens
        Pad(axis=0, pad_val=ignore_label, dtype='int64') # labels
    ): fn(list(map(trans_func, samples)))

    batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=args.batch_size,
        collate_fn=batchify_fn)
    # test_loader = paddle.io.DataLoader(
    #     dataset=test_ds,
    #     batch_size=args.batch_size,
    #     collate_fn=batchify_fn)

    num_training_steps = len(train_loader) * args.num_epoch
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

    metric = ChunkEvaluator(label_list=train_ds.label_vocab.keys(), suffix=False)
    criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    step, best_f1 = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, seq_lens, labels) in enumerate(train_loader):
            logits = model(input_ids, token_type_ids).reshape(
                [-1, train_ds.label_num])
            loss = paddle.mean(criterion(logits, labels.reshape([-1])))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                p, r, f1, avg_loss = evaluate(model, criterion, metric, len(label_map), dev_loader)
                print(f'dev step: {step} - loss: {avg_loss:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                        f'f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    best_f1 = f1
                    print(f'==============================================save best model ' \
                            f'best performerence {best_f1:5f}')
                    paddle.save(model.state_dict(), '{}/{}.best.pdparams'.format(args.checkpoints, task_type))
            step += 1

    # save the final model
    if rank == 0:
        paddle.save(model.state_dict(), '{}/{}.final.pdparams'.format(args.checkpoints, task_type))


def do_predict(args, task_type):
    paddle.set_device(args.device)
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    trigger_dict, role_dict = schema_loader(args.tag_path)
    if task_type == 'trigger':
        label_map = trigger_dict
        pretrained_model_path = args.init_trigger_ckpt
    elif task_type == 'argument':
        label_map = role_dict
        pretrained_model_path = args.init_argument_ckpt
    model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))
    id2label = {val: key for key, val in label_map.items()}
    no_entity_label = "O"
    ignore_label = len(label_map)

    print("============start predict==========")
    if not pretrained_model_path or not os.path.isfile(pretrained_model_path):
        raise Exception("init checkpoints {} not exist".format(pretrained_model_path))
    else:
        state_dict = paddle.load(pretrained_model_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % pretrained_model_path)

    # load data from predict file
    if task_type == 'trigger':
        pred_data = args.predict_data
    elif task_type == 'argument':
        pred_data = args.predict_temp_save_path
    else:
        assert False, "wrong task type"
    sentences, segments = predict_data_dealer(pred_data, task_type)
    encoded_inputs_list = []
    for sent, segment in zip(sentences, segments):
        input_ids, token_type_ids, seq_len = convert_example_to_feature([list(sent), [], segment], tokenizer,
                    max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'), # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'), # token_type_ids
        Stack(dtype='int64') # sequence lens
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [encoded_inputs_list[i: i + args.batch_size]
                            for i in range(0, len(encoded_inputs_list), args.batch_size)]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=-1)
        probs_ids = paddle.argmax(probs, -1).numpy()
        probs = probs.numpy()
        for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(), seq_lens.tolist()):
            prob_one = [p_list[index][pid] for index, pid in enumerate(p_ids[1: seq_len - 1])]
            label_one = [id2label[pid] for pid in p_ids[1: seq_len - 1]]
            results.append({"probs": prob_one, "labels": label_one})
    assert len(results) == len(sentences)
    print("============start write==========")
    write_data = []
    if task_type == 'trigger':
        for sent, ret in zip(sentences, results):
            temp = {'text':sent, 'event_list': []}
            event_list = pred2formot(ret['labels'], sent)
            for item in event_list:
                trigger = {"event_type": item[0], "trigger": item[3], "trigger_start_index": item[1], "arguments":[]}
                temp['event_list'].append(trigger)
            write_data.append(temp)
        write_data = [json.dumps(sent, ensure_ascii=False) for sent in write_data]
        with open(args.predict_temp_save_path, "w") as outfile:
            [outfile.write(d + "\n") for d in write_data]
        print("save data {} to {}".format(len(sentences), args.predict_temp_save_path))
    elif task_type == 'argument':
        trigger_data = [json.loads(line.strip()) for line in open(args.predict_temp_save_path, 'r', encoding='utf-8').readlines()]
        count = 0
        for item in trigger_data:
            for sent, ret, segment in zip(sentences, results, segments):
                if item['text'] == sent:
                    count += 1
                    for event in item['event_list']:
                        if segment[event['trigger_start_index']] == 1 and segment[event['trigger_start_index'] + len(event['trigger']) - 1] == 1:
                            arguments_list = pred2formot(ret['labels'], sent)
                            for argument_item in arguments_list:
                                argument = {"role": argument_item[0], "argument": argument_item[3], "argument_start_index": argument_item[1]}
                                event['arguments'].append(argument)
            write_data.append(item)
        write_data = [json.dumps(sent, ensure_ascii=False) for sent in write_data]
        with open(args.predict_save_path, "w") as outfile:
            [outfile.write(d + "\n") for d in write_data]
        print("save data {} to {}".format(len(sentences), args.predict_save_path))
        print(count)
    else:
        assert False, "wrong task type"

@Event_ExtractionModel.register("Event_Extraction","Paddle")
class Event_ExtractionPaddle(Event_ExtractionModel):
    '''Base class for event extract trainer'''

    def __init__(self, args, name: str = 'Event_ExtractionModel'):
        self.name = name
        self.args = args
    #two seqence tagging task: trigger detection and argument extraction

    def run(self):
        do_train(self.args, task_type='trigger')
        do_train(self.args, task_type='argument')

    def pred(self):
        do_predict(self.args, task_type='trigger')
        do_predict(self.args, task_type='argument')
