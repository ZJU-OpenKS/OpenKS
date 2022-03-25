from functools import partial
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as functional

import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from ..model import HypernymDiscoveryModel


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed
    to be used in a sequence-pair classification task.

    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        max_seq_length:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["hypo"], text_pair=example["hyper"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read(file_path):  # 1 表示是上下位关系，0表示不是上下位关系
    with open(file_path, 'r', encoding='utf-8') as f_in:
        for cnt, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            res_lst = line.split('\t')
            if len(res_lst) != 3:
                print('数据集错误，行数%d' % cnt)
                return None
            hypo, hyper, label = res_lst
            hypo, hyper = hypo.replace('_', ''), hyper.replace('_', '')  # 去掉数据集中的下划线
            label = 1 if label == 'hyper' else 0

            yield {'hypo': hypo, 'hyper': hyper, 'label': label}


@HypernymDiscoveryModel.register("HypernymDiscovery", "Paddle")
class HypernymDiscoveryPaddle(HypernymDiscoveryModel):
    def __init__(self, args, name: str = 'HypernymDiscoveryModel', ):
        super().__init__()
        self.name = name
        self.args = args

    @paddle.no_grad()
    def evaluate(self, model, criterion, metric, data_loader):
        """
        Given a dataset, it evals model and computes the metric.

        Args:
            model(obj:`paddle.nn.Layer`): A model to classify texts.
            data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
            criterion(obj:`paddle.nn.Layer`): It can compute the loss.
            metric(obj:`paddle.metric.Metric`): The evaluation metric.
        """
        model.eval()
        metric.reset()
        losses = []
        for batch in data_loader:
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            probs = functional.softmax(logits, axis=1)

            loss = criterion(probs, labels)

            losses.append(loss.numpy())

            correct = metric.compute(probs, labels)
            metric.update(correct)
            accu = metric.accumulate()

        print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
        model.train()
        metric.reset()

    def run(self):
        args = self.args
        paddle.set_device(args.device)
        rank = paddle.distributed.get_rank()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

        set_seed(args.seed)

        train_ds = load_dataset(read, file_path=args.train_path, lazy=False)
        dev_ds = load_dataset(read, file_path=args.dev_path, lazy=False)

        model = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_class=2)
        tokenizer = paddlenlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese')

        trans_func = partial(
            convert_example,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
            Stack(dtype="int64")  # label
        ): [data for data in fn(samples)]
        train_data_loader = create_dataloader(
            train_ds,
            mode='train',
            batch_size=args.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)
        dev_data_loader = create_dataloader(
            dev_ds,
            mode='dev',
            batch_size=args.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)

        if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
            state_dict = paddle.load(args.init_from_ckpt)
            model.set_dict(state_dict)
        model = paddle.DataParallel(model)

        num_training_steps = len(train_data_loader) * args.epochs

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                             args.warmup_proportion)

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

        criterion = paddle.nn.loss.CrossEntropyLoss()
        metric = paddle.metric.Accuracy()

        global_step = 0
        tic_train = time.time()
        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch
                logits = model(input_ids, token_type_ids)
                probs = functional.softmax(logits, axis=1)
                loss = criterion(probs, labels)

                correct = metric.compute(loss, labels)  # labels可以是索引，也可以是独热表示
                metric.update(correct)  # 更新正确预测的个数以及总个数
                acc = metric.accumulate()

                global_step += 1
                if global_step % 10 == 0 and rank == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, acc,
                           10 / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % 100 == 0 and rank == 0:  # 100步保存一次模型
                    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    self.evaluate(model, criterion, metric, dev_data_loader)
                    model._layers.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
