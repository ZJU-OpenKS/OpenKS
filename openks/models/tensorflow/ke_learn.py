# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import argparse
import tensorflow as tf
import numpy as np
import ast
import os
from ..model import KELearnModel
from .utils import extract_kvpairs_in_bio, cal_f1_score, cal_f1_score_org_pro, load_vocabulary
from .utils import DataProcessor_LSTM as DataProcessor
from .utils import DataProcessor_LSTM_for_sentences as DataProcessor_predict
from ..model import logger

@KELearnModel.register("KELearn", "TensorFlow")
class KELearnTorch(KELearnModel):
    def __init__(self, name='tensorflow-default', dataset=None, model=None, args=None):
        self.name = name
        self.dataset = dataset
        self.args = args
        self.model = model

        self.train_text_data = []
        self.train_label_data = []
        self.test_text_data = []
        self.test_label_data = []
        self.w2i_char = {}
        self.i2w_char = {}
        self.w2i_bio = {}
        self.i2w_bio = {}

        words = set()
        labels = set()
        train_set = self.dataset.bodies[0]
        valid_set = self.dataset.bodies[1]
        for sentence in train_set:
            word_list = ast.literal_eval(sentence[0])
            label_list = ast.literal_eval(sentence[1])
            assert(len(word_list) == len(label_list))
            self.train_text_data.append(word_list)
            self.train_label_data.append(label_list)
            words.update(word_list)
            labels.update(label_list)
        print("Get {} train sentences!".format(len(self.train_text_data)))
        for sentence in valid_set:
            word_list = ast.literal_eval(sentence[0])
            label_list = ast.literal_eval(sentence[1])
            assert(len(word_list) == len(label_list))
            self.test_text_data.append(word_list)
            self.test_label_data.append(label_list)
            words.update(word_list)
            labels.update(label_list)
        words = list(words)
        labels = list(labels)
        words = sorted(words)
        labels = sorted(labels)
        words.insert(0, "[PAD]")
        words.insert(1, "[UNK]")
        words.insert(2, "[SEP]")
        words.insert(3, "[SPA]")
        print("Get {} valid sentences!".format(len(self.test_text_data)))

        logger.info("loading vocab...")

        self.w2i_char, self.i2w_char = load_vocabulary(words)
        self.w2i_bio, self.i2w_bio = load_vocabulary(labels)

    def predict(self, text):
        # load model
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        model = self.model(
            embedding_dim=self.args["embedding_dim"],
            hidden_dim=self.args["hidden_dim"],
            vocab_size_char=len(self.w2i_char),
            vocab_size_bio=len(self.w2i_bio),
            use_crf=self.args["use_crf"]
        )

        sess = tf.Session(config=tf_config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.args["model_dir"]))

        data_processor_valid = DataProcessor_predict(
            text,
            self.w2i_char,
            self.w2i_bio, 
            shuffling=False
        )
        result = []

        def valid(data_processor, max_batches=None, batch_size=1024):
            preds_kvpair = []
            golds_kvpair = []
            batches_sample = 0

            while True:
                (inputs_seq_batch, 
                    inputs_seq_len_batch,
                    outputs_seq_batch) = data_processor.get_batch(batch_size)

                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.outputs_seq: outputs_seq_batch
                }

                preds_seq_batch = sess.run(model.outputs, feed_dict)
                
                for pred_seq, gold_seq, input_seq, l in zip(preds_seq_batch, 
                                                            outputs_seq_batch, 
                                                            inputs_seq_batch, 
                                                            inputs_seq_len_batch):
                    pred_seq = [self.i2w_bio[i] for i in pred_seq[:l]]
                    gold_seq = [self.i2w_bio[i] for i in gold_seq[:l]]
                    char_seq = [self.i2w_char[i] for i in input_seq[:l]]
                    pred_kvpair = extract_kvpairs_in_bio(pred_seq, char_seq)
                    gold_kvpair = extract_kvpairs_in_bio(gold_seq, char_seq)
                    
                    preds_kvpair.append(pred_kvpair)
                    golds_kvpair.append(gold_kvpair)
                    
                if data_processor.end_flag:
                    data_processor.refresh()
                    break
                
                batches_sample += 1
                if (max_batches is not None) and (batches_sample >= max_batches):
                    break

            for item in preds_kvpair:
                ner_dic = {}
                for t in item:
                    ner_dic[t[1]] = t[0]
                result.append(ner_dic)
            return

        valid(data_processor_valid, max_batches=1)
        return result

    def run(self):
        # logging
        # set logging
        """
        if not os.path.exists(self.args["model_dir"]):
            os.mkdir(self.args["model_dir"])
        log_file_path = self.args["log_file_path"]
        if os.path.exists(log_file_path): os.remove(log_file_path)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        fhlr = logging.FileHandler(log_file_path)
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        """

        logger.info("loading data...")

        data_processor_train = DataProcessor(
            self.train_text_data,
            self.train_label_data,
            self.w2i_char,
            self.w2i_bio, 
            shuffling=True
        )

        data_processor_valid = DataProcessor(
            self.test_text_data,
            self.test_label_data,
            self.w2i_char,
            self.w2i_bio, 
            shuffling=True
        )

        logger.info("building model...")

        model = self.model(
            embedding_dim=self.args["embedding_dim"],
            hidden_dim=self.args["hidden_dim"],
            vocab_size_char=len(self.w2i_char),
            vocab_size_bio=len(self.w2i_bio),
            use_crf=self.args["use_crf"]
        )

        logger.info("model params:")
        params_num_all = 0
        for variable in tf.trainable_variables():
            params_num = 1
            for dim in variable.shape:
                params_num *= dim
            params_num_all += params_num
            logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
        logger.info("all params num: " + str(params_num_all))
                
        logger.info("start training...")

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=50)

            epoches = 0
            losses = []
            batches = 0
            best_f1 = 0
            batch_size = 32

            while epoches < 20:
                (inputs_seq_batch, 
                inputs_seq_len_batch,
                outputs_seq_batch) = data_processor_train.get_batch(batch_size)
                
                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.outputs_seq: outputs_seq_batch
                }
                
                if batches == 0: 
                    logger.info("###### shape of a batch #######")
                    logger.info("input_seq: " + str(inputs_seq_batch.shape))
                    logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
                    logger.info("output_seq: " + str(outputs_seq_batch.shape))
                    logger.info("###### preview a sample #######")
                    logger.info("input_seq:" + " ".join([self.i2w_char[i] for i in inputs_seq_batch[0]]))
                    logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
                    logger.info("output_seq: " + " ".join([self.i2w_bio[i] for i in outputs_seq_batch[0]]))
                    logger.info("###############################")
                
                loss, _ = sess.run([model.loss, model.train_op], feed_dict)
                losses.append(loss)
                batches += 1
                
                if data_processor_train.end_flag:
                    data_processor_train.refresh()
                    epoches += 1
        
                def valid(data_processor, max_batches=None, batch_size=1024):
                    preds_kvpair = []
                    golds_kvpair = []
                    batches_sample = 0
                    
                    while True:
                        (inputs_seq_batch, 
                        inputs_seq_len_batch,
                        outputs_seq_batch) = data_processor.get_batch(batch_size)

                        feed_dict = {
                            model.inputs_seq: inputs_seq_batch,
                            model.inputs_seq_len: inputs_seq_len_batch,
                            model.outputs_seq: outputs_seq_batch
                        }

                        preds_seq_batch = sess.run(model.outputs, feed_dict)
                        
                        for pred_seq, gold_seq, input_seq, l in zip(preds_seq_batch, 
                                                                    outputs_seq_batch, 
                                                                    inputs_seq_batch, 
                                                                    inputs_seq_len_batch):
                            pred_seq = [self.i2w_bio[i] for i in pred_seq[:l]]
                            gold_seq = [self.i2w_bio[i] for i in gold_seq[:l]]
                            char_seq = [self.i2w_char[i] for i in input_seq[:l]]
                            pred_kvpair = extract_kvpairs_in_bio(pred_seq, char_seq)
                            gold_kvpair = extract_kvpairs_in_bio(gold_seq, char_seq)
                            
                            preds_kvpair.append(pred_kvpair)
                            golds_kvpair.append(gold_kvpair)
                            
                        if data_processor.end_flag:
                            data_processor.refresh()
                            break
                        
                        batches_sample += 1
                        if (max_batches is not None) and (batches_sample >= max_batches):
                            break
                    
                    p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)
                    p_org, r_org, f1_org, p_pro, r_pro, f1_pro = cal_f1_score_org_pro(preds_kvpair, golds_kvpair)
                    
                    logger.info("Valid Samples: {}".format(len(preds_kvpair)))
                    logger.info("Valid P/R/F1: {} / {} / {}".format(round(p*100, 2), round(r*100, 2), round(f1*100, 2)))
                    logger.info("Valid ORG P/R/F1: {} / {} / {}".format(round(p_org*100, 2), round(r_org*100, 2), round(f1_org*100, 2)))
                    logger.info("Valid PRO P/R/F1: {} / {} / {}".format(round(p_pro*100, 2), round(r_pro*100, 2), round(f1_pro*100, 2)))

                    return (p, r, f1)
                    
                if batches % 100 == 0:
                    logger.info("")
                    logger.info("Epoches: {}".format(epoches))
                    logger.info("Batches: {}".format(batches))
                    logger.info("Loss: {}".format(sum(losses) / len(losses)))
                    losses = []
                    
                    p, r, f1 = valid(data_processor_valid, max_batches=10)
                    if f1 > best_f1:
                        best_f1 = f1
                        ckpt_save_path = self.args["model_dir"] + "model.ckpt".format(batches)
                        logger.info("Path of ckpt: {}".format(ckpt_save_path))
                        saver.save(sess, ckpt_save_path)
                        logger.info("############# best performance now here ###############")
                    
