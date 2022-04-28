# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import os
import json
import random
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, BertTokenizer, BertModel
from paddlenlp.metrics import ChunkEvaluator
from ..model import Causality_ExtractionModel
#from utils import read_by_lines, write_by_lines, load_dict



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
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']


    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len-2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        return input_ids, token_type_ids, seq_len, encoded_label

def schema_loader():
    CE_dict = {'O': 0}
    CE_dict['B-C'] = 1
    CE_dict['I-C'] = 2
    CE_dict['B-E'] = 3
    CE_dict['I-E'] = 4
    CE_dict['B-Emb'] = 5
    CE_dict['I-Emb'] = 6

    return CE_dict

def data_dealer(path):
    corpus = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8').readlines()]
    text_list = []
    label_ids_list = []
    for item in corpus:
        text = item['words']
        label_ids = ['O'] * len(text)
        cause_span_list = []
        effect_span_list = []
        both_span_list = []
        for causal_triple in item['causal_triples']:
            cause_span_list.append(causal_triple[0])
            effect_span_list.append(causal_triple[1])
        for i in range(len(cause_span_list)-1, -1, -1):
            if cause_span_list[i] in effect_span_list:
                both_span_list.append(cause_span_list[i])
                effect_span_list.remove(cause_span_list[i])
                cause_span_list.remove(cause_span_list[i])

        for item in cause_span_list:
            label_ids[item[0]] = "B-C"
            for i in range(item[0] + 1, item[1]):
                label_ids[i] = "I-C"

        for item in effect_span_list:
            label_ids[item[0]] = "B-E"
            for i in range(item[0] + 1, item[1]):
                label_ids[i] = "I-E"

        for item in both_span_list:
            label_ids[item[0]] = "B-Emb"
            for i in range(item[0] + 1, item[1]):
                label_ids[i] = "I-Emb"

        text_list.append(text)
        label_ids_list.append(label_ids)

    return text_list, label_ids_list

def predict_data_dealer(path):
    corpus = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8').readlines()]
    text_list = []
    for item in corpus:
        text = item['words']
        text_list.append(text)
    return text_list

def MLP_data_dealer(path):
    corpus = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8').readlines()]
    text_list = []
    causality_pair_list = []
    for item in corpus:
        text = item['words']
        causality_pair_list.append(item['causal_triples'])
        text_list.append(text)
    return text_list, causality_pair_list

def MLP_pred_line_data_generator(text, causality_pair, model, tokenizer):
    pred_data_list = []
    count = 0
    if len(causality_pair) == 0:
        return []
    vec = tokenizer.encode([t.lower() for t in text])['input_ids']
    vec2 = paddle.to_tensor([vec], dtype='int64')
    bert_emb = model(vec2)
    cause_span_list = []
    effect_span_list = []
    count += 1
    for item in causality_pair:
        cause_span_list.append(item[0])
        effect_span_list.append(item[1])
        c_span = bert_emb[0][:, item[0][0] + 1:item[0][1] + 1, :]
        e_span = bert_emb[0][:, item[1][0] + 1:item[1][1] + 1, :]
        C_vector = paddle.mean(c_span, axis=1)
        E_vector = paddle.mean(e_span, axis=1)
        pred_data_list.append(paddle.concat(x=[C_vector, E_vector], axis=1).tolist())
    return pred_data_list

def MLP_data_generator(text_list, causality_pair_list, model, tokenizer):
    positive_example = []
    negativa_example = []
    count = 0
    for text, causality_pair in zip(text_list, causality_pair_list):
        if len(causality_pair) == 0:
            continue
        vec = tokenizer.encode([t.lower() for t in text])['input_ids']
        vec2 = paddle.to_tensor([vec], dtype='int64')
        bert_emb = model(vec2)
        cause_span_list = []
        effect_span_list = []
        cause_vector_list = []
        effect_vector_list = []
        count += 1
        for item in causality_pair:
            cause_span_list.append(item[0])
            effect_span_list.append(item[1])
            c_span = bert_emb[0][:, item[0][0] + 1:item[0][1] + 1, :]
            e_span = bert_emb[0][:, item[1][0] + 1:item[1][1] + 1, :]
            C_vector = paddle.mean(c_span, axis=1)
            E_vector = paddle.mean(e_span, axis=1)
            cause_vector_list.append(C_vector)
            effect_vector_list.append(E_vector)
            positive_example.append(paddle.concat(x=[C_vector, E_vector], axis=1).tolist())
            negativa_example.append(paddle.concat(x=[E_vector, C_vector], axis=1).tolist())
            for C_item, C_vec in zip(cause_span_list, cause_vector_list):
                if len(negativa_example) > 2 * len(positive_example):
                    break
                for E_item, E_vec in zip(effect_span_list , effect_vector_list):
                    flag = True
                    for item in causality_pair:
                        if C_item == item[0] and E_item == item[1]:
                            flag = False
                    if flag:
                        print((C_item,E_item))
                        print(causality_pair)
                        negativa_example.append(paddle.concat(x=[C_vec, E_vec], axis=1).tolist())
                        if len(negativa_example) > 2 * len(positive_example):
                            break
        print(count)
    return positive_example, negativa_example


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

class CausalityExtraction(paddle.io.Dataset):
    """DuEventExtraction"""
    def __init__(self, data_path):
        causality_dict = schema_loader()

        self.label_vocab = causality_dict
        text_list, label_ids_list = data_dealer(data_path)
        self.word_ids = []
        self.label_ids = []
        for item in text_list:
            words = [t.lower() for t in item]
            self.word_ids.append(words)
        self.label_ids = label_ids_list
        self.label_num = len(self.label_vocab)

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]

class CausalityMatch(paddle.io.Dataset):
    """DuEventExtraction"""
    def __init__(self, data_path):
        causality_dict = schema_loader()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_vocab = causality_dict
        text_list, causality_pair_list = MLP_data_dealer(data_path)
        positive_example, negativa_example = MLP_data_generator(text_list, causality_pair_list, self.bert_model, self.tokenizer)
        dealed_data = []
        label_data = []
        dealed_data.extend(positive_example)
        label_data.extend([1]*len(positive_example))
        dealed_data.extend(negativa_example)
        label_data.extend([0]*len(negativa_example))
        self.word_ids = [t[0] for t in dealed_data]
        self.label_ids = label_data

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return paddle.to_tensor(self.word_ids[index], dtype='float32'), paddle.to_tensor(self.label_ids[index], dtype='int64')

class MLP_Model(nn.Layer):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(768*2, 96),
            nn.ReLU(),
            paddle.nn.Dropout(0.2),
            nn.Linear(96, 2)
        )

    def forward(self, inputs):
        y = self.MLP(inputs)
        return y

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
    PLM_train_dataset = CausalityMatch(args.train_data)
    PLM_test_dataset = CausalityMatch(args.test_data)

    #----------------------MLP:if cause and effect match--------------------------------------------------------------
    MLP_model = paddle.Model(MLP_Model())

    MLP_model.prepare(paddle.optimizer.AdamW(parameters=MLP_model.parameters()),
                      paddle.nn.CrossEntropyLoss(),
                      paddle.metric.Accuracy())

    MLP_model.fit(PLM_train_dataset,
                  PLM_test_dataset,
                  epochs=5,
                  batch_size=32,
                  verbose=1,
                  save_dir=args.MLP_save_path)

    MLP_model.evaluate(PLM_test_dataset, verbose=1)

    #---------------------ernie:to find cause and effect--------------------------------------------------------------
    no_entity_label = "O"
    ignore_label = -1

    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    causality_dict = schema_loader()
    label_map = causality_dict

    model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))
    model = paddle.DataParallel(model)



    print("============start train==========")
    train_ds = CausalityExtraction(args.train_data)
    test_ds = CausalityExtraction(args.test_data)

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
    test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        collate_fn=batchify_fn)


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
                p, r, f1, avg_loss = evaluate(model, criterion, metric, len(label_map), test_loader)
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
    causality_dict = schema_loader()
    label_map = causality_dict
    pretrained_model_path = os.path.join(args.init_ckpt, "Erine/best.pdparams")
    MLP_path = os.path.join(args.init_ckpt, "MLP")

    model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))
    id2label = {val: key for key, val in label_map.items()}
    no_entity_label = "O"
    ignore_label = len(label_map)

    MLP_model = MLP_Model()


    print("============model loading==========")
    if not pretrained_model_path or not os.path.isfile(pretrained_model_path):
        raise Exception("init checkpoints {} not exist".format(pretrained_model_path))
    else:
        state_dict = paddle.load(pretrained_model_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % pretrained_model_path)

    if not MLP_path or not os.path.isdir(MLP_path):
        raise Exception("init checkpoints {} not exist".format(MLP_path))
    else:
        MLP_state = paddle.load(os.path.join(MLP_path, 'final.pdparams'))
        MLP_model.set_dict(MLP_state)
    MLP_model.eval()
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # load data from predict file
    pred_data = args.predict_data

    sentences = predict_data_dealer(pred_data)
    encoded_inputs_list = []
    for sent in sentences:
        input_ids, token_type_ids, seq_len = convert_example_to_feature([sent, []], tokenizer,
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
        probs = F.softmax(logits, axis=-1)
        probs_ids = paddle.argmax(probs, -1).numpy()
        probs = probs.numpy()
        for p_list, p_ids, seq_len, sentence in zip(probs.tolist(), probs_ids.tolist(), seq_lens.tolist(), sentence_batch):
            prob_one = [p_list[index][pid] for index, pid in enumerate(p_ids[1: seq_len - 1])]
            label_one = [id2label[pid] for pid in p_ids[1: seq_len - 1]]
            temp_list = pred2formot(label_one, sentence)
            pred_causality_pair_list = []
            possible_causality_pair_list = []
            for i in range(0, len(temp_list)):
                for j in range(i + 1, len(temp_list)):
                    if (temp_list[i][0] == 'C' and (temp_list[j][0] =='E' or temp_list[j][0] == 'Emb')) or ((temp_list[i][0] == 'C' or temp_list[i][0] == 'Emb')and temp_list[j][0] == 'E'):
                        possible_causality_pair_list.append(
                            [[temp_list[i][1],temp_list[i][2]],[temp_list[j][1],temp_list[j][2]]])
                    if (temp_list[i][0] == 'E' and (temp_list[j][0] == 'C' or temp_list[j][0] == 'Emb')) or (
                            (temp_list[i][0] == 'E' or temp_list[i][0] == 'Emb') and temp_list[j][0] == 'C'):
                        possible_causality_pair_list.append(
                            [[temp_list[j][1], temp_list[j][2]], [temp_list[i][1], temp_list[i][2]]])
            pred_pair_data_list = MLP_pred_line_data_generator(sentence, possible_causality_pair_list, bert_model, bert_tokenizer)
            for m in range(0, len(pred_pair_data_list)):
                if_cause = MLP_model(paddle.to_tensor(pred_pair_data_list[m]))
                if int(paddle.argmax(if_cause, axis=1)) == 1:
                    pred_causality_pair_list.append(possible_causality_pair_list[m])
            results.append({"words": list(sentence), "causal_triples": pred_causality_pair_list})
    with open(args.predict_save_path, 'w') as pred_ouput_file:
        for item in results:
            pred_ouput_file.write(json.dumps(item) + '\n')


@Causality_ExtractionModel.register("Causality_Extraction","Paddle")
class Causality_ExtractionPaddle(Causality_ExtractionModel):
    '''Base class for event extract trainer'''

    def __init__(self, args, name: str = 'Event_ExtractionModel'):
        self.name = name
        self.args = args

    def run(self):
        do_train(self.args)

    def pred(self):
        do_predict(self.args)

