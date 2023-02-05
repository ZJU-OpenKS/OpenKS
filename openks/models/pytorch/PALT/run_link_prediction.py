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
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger.setLevel(level=logging.DEBUG)
    file_handler = logging.FileHandler("%s/%s-test-log-%s-%s.txt" % \
        (os.getenv("EXP_ROOT"), data_args.filename_info, str(data_args.test_count), str(data_args.test_worker_id)))
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16
    )
    logger.info('Training/evaluation parameters %s' % training_args)
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
        prediction_begin_time = time.time()
        #trainer.model.set_predict_mode()
        trainer.prediction_loss_only = False
        trainer.data_collator.set_predict_mode()
        train_triples = processor.get_train_triples()
        dev_triples = processor.get_dev_triples()
        test_triples = processor.get_test_triples()
        all_triples = train_triples + dev_triples + test_triples
        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)
        ranks = []
        ranks_left = []
        ranks_right = []
        hits_left = []
        hits_right = []
        hits = []
        top_ten_hit_count = 0
        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])
        total_test = len(test_triples)
        for (test_id, test_triple) in enumerate(test_triples):
            if data_args.test_count is not None:
                assert data_args.test_worker_id is not None
                assert data_args.test_worker_id < data_args.test_count
                if test_id % data_args.test_count != data_args.test_worker_id:
                    continue
            if np.random.random() > data_args.test_ratio:
                continue
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]
            head_corrupt_list = [test_triple]
            if data_args.type_constrain:
                tmp_entity_list = processor.rel2valid_head[relation]
            else:
                tmp_entity_list = processor.get_entities()
            for corrupt_ent in tmp_entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        head_corrupt_list.append(tmp_triple)
            (_, tmp_features) = processor._create_examples_and_features(head_corrupt_list)
            data_len = len(tmp_features)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)
            all_pos_indicator = torch.tensor([f.pos_indicator for f in tmp_features], dtype=torch.long)
            eval_data = DictDataset(input_ids=all_input_ids, labels=all_label_ids, pos_indicator=all_pos_indicator)
            trainer.data_collator.predict_mask_part = 0
            preds = trainer.predict(eval_data).predictions
            preds = torch.tensor(preds)
            shape = preds.shape
            if len(shape) > 1 and shape[1] > 1:
                preds = torch.nn.functional.softmax(preds)[:, 0]
            if trainer.is_world_master():
                argsort1 = np.argsort(-preds)
                rank1 = np.where(argsort1 == 0)[0][0]
                logger.info('left: ' + str(rank1) + str(data_len))
                ranks.append(rank1 + 1)
                ranks_left.append(rank1 + 1)
                if rank1 < 10:
                    top_ten_hit_count += 1
            tail_corrupt_list = [test_triple]
            if data_args.type_constrain:
                tmp_entity_list = processor.rel2valid_tail[relation]
            else:
                tmp_entity_list = processor.get_entities()
            for corrupt_ent in tmp_entity_list:
                if corrupt_ent != tail:
                    tmp_triple = [head, relation, corrupt_ent]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        tail_corrupt_list.append(tmp_triple)
            (_, tmp_features) = processor._create_examples_and_features(tail_corrupt_list)
            data_len = len(tmp_features)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)
            all_pos_indicator = torch.tensor([f.pos_indicator for f in tmp_features], dtype=torch.long)
            eval_data = DictDataset(input_ids=all_input_ids, labels=all_label_ids, pos_indicator=all_pos_indicator)
            trainer.data_collator.predict_mask_part = 2
            preds = trainer.predict(eval_data).predictions
            preds = torch.tensor(preds)
            shape = preds.shape
            if len(shape) > 1 and shape[1] > 1:
                preds = torch.nn.functional.softmax(preds)[:, 0]
            if trainer.is_world_master():
                argsort1 = np.argsort(-preds)
                rank2 = np.where(argsort1 == 0)[0][0]
                ranks.append(rank2 + 1)
                ranks_right.append(rank2 + 1)
                logger.info('right: ' + str(rank2) + str(data_len))
                logger.info('mean rank until now: ' + str(np.mean(ranks)))
                if rank2 < 10:
                    top_ten_hit_count += 1
                logger.info('hit@10 until now: ' + str(top_ten_hit_count * 1.0 / len(ranks)))
                logger.info('time used for prediction now: ' + str(time.time() - prediction_begin_time))
                logger.info('num of tested triples: {} / {}'.format(test_id + 1, total_test))
                for hits_level in range(10):
                    if rank1 <= hits_level:
                        hits[hits_level].append(1.0)
                        hits_left[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        hits_left[hits_level].append(0.0)
                    if rank2 <= hits_level:
                        hits[hits_level].append(1.0)
                        hits_right[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        hits_right[hits_level].append(0.0)
        if trainer.is_world_master():
            for i in [0, 2, 9]:
                logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
                logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
                logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
            logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
            logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
            logger.info('Mean rank: {0}'.format(np.mean(ranks)))
            logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1.0 / np.array(ranks_left))))
            logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1.0 / np.array(ranks_right))))
            logger.info('Mean reciprocal rank: {0}'.format(np.mean(1.0 / np.array(ranks))))
    print(training_args.output_dir)

if __name__ == '__main__':
    main()
