import logging
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from transformers import BertForSequenceClassification, BertForNextSentencePrediction

from model.trainer import KGCTrainer
from model.data_processor import DictDataset, KGProcessor
from model.data_collator import PoolingCollator, PromptCollator, TempCollator
from model.utils import DataArguments, ModelArguments
from model.bert_template_model import PTuneNSP
import pickle

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == 'kg':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    print(training_args.learning_rate)
    print(model_args.top_layer_nums)
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and (not training_args.overwrite_output_dir)
    ):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN
    )
    assert (model_args.word_embedding_type in [None, "linear", "double-linear", "mlp"])
    assert (model_args.top_additional_layer_type in [None, "linear", "double-linear", "mlp", "adapter-module"])


    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16
    )
    logger.info('Training/evaluation parameters %s', training_args)
    set_seed(training_args.seed)
    if training_args.do_train and not model_args.load_checkpoint:
        use_cache = True
        logger.info("Using cached pretrained model.")
    elif model_args.checkpoint_dir is None:
        use_cache = True
        logger.info("Zero shot prediction.")
    else:
        use_cache = False
        logger.info("In prediction setting or using checkpoint.")
        model_checkpoint = model_args.checkpoint_dir
        if model_args.model_checkpoint_num:
            print("Using checkpoint num: %d." % model_args.model_checkpoint_num)
            model_checkpoint += "/checkpoint-%d" % model_args.model_checkpoint_num
        logger.info("Using checkpoint from dir %s" % model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path if use_cache else model_args.checkpoint_dir,
        cache_dir=model_args.model_cache_dir
    )
    is_world_process_zero = training_args.local_rank == -1 or torch.distributed.get_rank() == 0
    processor = KGProcessor(data_args, tokenizer, is_world_process_zero)
    (train_data, dev_data, test_data) = processor.get_dataset(training_args)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir
    )
    if not hasattr(config, 'real_vocab_size'):
        config.real_vocab_size = config.vocab_size
    if model_args.pos_weight is not None:
        model_args.pos_weight = torch.tensor([model_args.pos_weight]).to(training_args.device)
    if model_args.model_type == "template":
        if use_cache:
            tokenizer.add_special_tokens({'additional_special_tokens': [data_args.pseudo_token]})
        pseudo_token_id = tokenizer.convert_tokens_to_ids(data_args.pseudo_token)
        pad_token_id, unk_token_id = tokenizer.pad_token_id, tokenizer.unk_token_id
        template = [data_args.begin_temp, data_args.mid_temp, data_args.end_temp]
        print("="*10 + "using prompt model" + "="*10)
        if model_args.use_NSP:
            model = PTuneNSP.from_pretrained(
                model_args.model_name_or_path if use_cache else model_checkpoint,
                template=template,
                pseudo_token_id=pseudo_token_id,
                pad_token_id=pad_token_id,
                unk_token_id=unk_token_id,
                use_mlm_finetune=model_args.use_mlm_finetune,
                use_head_finetune=model_args.use_head_finetune,
                use_mlpencoder=model_args.use_mlpencoder,
                word_embedding_type=model_args.word_embedding_type,
                word_embedding_hidden_size=model_args.word_embedding_hidden_size,
                word_embedding_dropout=model_args.word_embedding_dropout,
                word_embedding_layernorm=model_args.word_embedding_layernorm,
                top_additional_layer_type=model_args.top_additional_layer_type,
                top_additional_layer_hidden_size=model_args.top_additional_layer_hidden_size,
                top_use_dropout=model_args.top_use_dropout,
                dropout_ratio=model_args.dropout_ratio,
                top_use_layernorm=model_args.top_use_layernorm,
                top_layer_nums=model_args.top_layer_nums,
                adapter_type=model_args.adapter_type,
                adapter_size=model_args.adapter_size,
                from_tf=bool('.ckpt' in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.model_cache_dir
            )
            if not model_args.not_print_model:
                print(model)
                with open("exp-result-record.txt", "a") as f:
                    print(model, file=f)
        else:
            raise NotImplementedError()
        data_collator = TempCollator(tokenizer, pseudo_token_id=pseudo_token_id,
                                     prompt_temp=template, nsp=model_args.use_NSP)
    elif model_args.model_type == "raw_bert":
        print("="*10 + "using bert model" + "="*10)
        if model_args.use_NSP:
            model = BertForNextSentencePrediction.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool('.ckpt' in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.model_cache_dir
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool('.ckpt' in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.model_cache_dir
            )
        data_collator = PromptCollator(tokenizer, nsp=model_args.use_NSP)
    else:
        raise NotImplementedError()
    trainer = KGCTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=dev_data,
        prediction_loss_only=True
    )
    if data_args.group_shuffle:
        print('using group shuffle')
        trainer.use_group_shuffle(data_args.num_neg)
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
    if training_args.do_predict:
        label_map = {'-1': 0, '1': 1}
        #trainer.model.set_predict_mode()
        trainer.prediction_loss_only = False
        trainer.data_collator.set_predict_mode()
        (dev_triples, dev_labels) = processor.get_dev_triples(return_label=True)
        dev_labels = np.array([label_map[l] for l in dev_labels], dtype=int)
        (_, tmp_features) = processor._create_examples_and_features(dev_triples)
        all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
        all_pos_indicator = torch.tensor([f.pos_indicator for f in tmp_features], dtype=torch.long)
        eval_data = DictDataset(input_ids=all_input_ids, pos_indicator=all_pos_indicator)
        trainer.data_collator.predict_mask_part = 0
        preds = trainer.predict(eval_data).predictions
        preds = torch.tensor(preds)
        shape = preds.shape
        if len(shape) > 1 and shape[1] > 1:
            preds = torch.nn.functional.softmax(preds)[:, 0].numpy()
        mean_dev = np.mean(preds)
        print('mean_dev: ', mean_dev)
        if len(shape) > 1 and shape[1] > 1:
            a, b = 0, 1
        else:
            a, b = -5, 5
        max_acc = 0
        for i in range(1000):
            m = (b - a) / 1000 * i + a
            tmp_preds = preds - m
            acc = np.mean((tmp_preds > 0).astype(int) == dev_labels)
            if acc > max_acc:
                max_acc = acc
                max_m = m
        print('max acc: ', max_acc)
        print('max m: ', max_m)
        mean_dev = max_m
        # mean_dev = 0.5

        (test_triples, test_labels) = processor.get_test_triples(return_label=True)
        test_labels = np.array([label_map[l] for l in test_labels], dtype=int)
        (_, tmp_features) = processor._create_examples_and_features(test_triples)
        all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
        all_pos_indicator = torch.tensor([f.pos_indicator for f in tmp_features], dtype=torch.long)
        eval_data = DictDataset(input_ids=all_input_ids, pos_indicator=all_pos_indicator)
        preds = trainer.predict(eval_data).predictions
        preds = torch.tensor(preds)
        shape = preds.shape
        if len(shape) > 1 and shape[1] > 1:
            preds = torch.nn.functional.softmax(preds)[:, 0].numpy()
        with open("case_study/%s-test_info.pkl" % model_args.checkpoint_dir.strip().split("/")[-1], "wb") as f:
            pickle.dump([mean_dev, preds], f)
        preds = preds - mean_dev
        acc = np.mean((preds > 0).astype(int) == test_labels)
        print('test acc: ', acc)
        with open("exp-result-record.txt", "a") as f:
            f.writelines([
                "Experiment: %s\n" % data_args.exp_info,
                "\tDev max acc: %.06f\n" % max_acc,
                "\tTest acc:    %.06f\n" % acc
            ])
    print(training_args.output_dir)


if __name__ == '__main__':
    main()
