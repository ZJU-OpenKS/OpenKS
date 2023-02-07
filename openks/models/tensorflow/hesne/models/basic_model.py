from typing import Tuple, List, Any, Sequence
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import json
from utils.data_manager import *
from models.hegraph import HeGraph

class BasicModel(object):
    @classmethod
    def default_params(cls):
        return{
            'num_epochs': 500,
            'hidden_size': 128,
            'learning_rate': 0.005, #0.0005 init
            'clamp_gradient_norm': 5.0,
            'eval_point': 1e2,
            'decay_step': 1e2,
            'decay': 0.98,
            'patience': 10,
            'node_type_numbers': 3,
            'batch_event_numbers': 64,
            'max_his_num': 50,
            'his_type': 'last',
            'aggregator_type': 'attention',
            'graph_rnn_cell': 'lstm',
            'graph_rnn_activation': 'tanh',
            'keep_prob': 0.8,
            'use_different_cell': True,
            'is_training': True
        }

    def __init__(self, args):
        self.args = args
        params = self.default_params()
        config_file = args.get('--config_file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--conifg')
        if config is not None:
            params.update((json.loads(config)))
        self.params = params
        data_dir = params['data_dir']
        log_dir = params['log_dir']
        save_dir = params['save_dir']
        data_set = params['dataset']
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.log_file = os.path.join(log_dir, self.get_log_file())
        self.checkpoint_dir = os.path.join(save_dir, self.get_checkpoint_dir())
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

# load train, valid, test data
        save_event_data(data_dir, self.params['batch_event_numbers'])
        self.train_data = load_event_data(os.path.join(data_dir, 'event_traindata.txt'), data_set)
        self.valid_data = load_event_data(os.path.join(data_dir, 'event_validdata.txt'), data_set)
        self.test_data = load_event_data(os.path.join(data_dir, 'event_testdata.txt'), data_set)
        self.degrees = get_degree(self.train_data, self.params['node_type_numbers'])
        print('load data done')
# build hegraph based on loaded data
        self.hegraph = HeGraph(self.params, self.train_data, self.valid_data, self.test_data)
        self.hegraph.build_hegraph()
        self.params['n_nodes_pertype'] = self.hegraph.get_totalnum_pertype()
        print('hegraph built')
        print(self.params['n_nodes_pertype'])
        self.train_data = BatchData(self.params, self.train_data, 0)
        self.valid_data = BatchData(self.params, self.valid_data, self.train_data.list_length)
        self.test_data = BatchData(self.params, self.test_data, self.train_data.list_length+self.valid_data.list_length)

# tensorflow config and graph build
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            if params['is_training']:
                self.make_train_step()
            # Restore/initialize variables:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            init_from = args.get('--init_from')
            if init_from is not None:
                self.params['init_from'] = init_from
                ckpt = tf.train.get_checkpoint_state(init_from)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('init from:'+init_from)
            else:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                print('Randomly initialize model')

    # def prepare_specific_graph_model(self) -> None:
    #     raise Exception("Models have to implement prepare_specific_graph_model!")
    #
    # def build_specific_graph_model(self, state) -> tf.Tensor:
    #     raise Exception("Models have to implement compute_final_node_representations!")

    def make_train_step(self):
        self.ops['lr'] = tf.maximum(1e-5, tf.train.exponential_decay(self.params['learning_rate'], \
                            self.ops['global_step'], self.params['decay_step'], self.params['decay'], staircase=True))
        optimizer = tf.train.AdamOptimizer(self.ops['lr'])
        # grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'])
        # print(tf.trainable_variables())
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_op'] = optimizer.apply_gradients(clipped_grads, global_step=self.ops['global_step'])
        print('make train step finished')

    def make_model(self):
        raise Exception("Models have to implement make_model!")

    def train(self):
        raise Exception("Models have to implement train!")

    def validation(self):
        raise Exception("Models have to implement valid!")

    def test(self):
        raise Exception("Models have to implement test!")


    # def test(self):
    #     self.test_data = BatchData(self.params, self.test_data)
    #     test_batches_num = self.test_data.get_batch_num()
    #     epoch_flag = False
    #     while not epoch_flag:
    #         fetches = [self.ops['embedding'], self.ops['celstates'], self.ops['predict']]
    #         # batch_feed_dict, epoch_flag = self.test_data.next_batch()
    #         batch_feed_dict = self.get_batch_feed_dict('test')
    #         self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
    #         self.node_embedding_cur, self.node_cellstates_cur, predict = self.sess.run(fetches, feed_dict=batch_feed_dict)
    #         res = evaluate_score()


            # test_batches_num, test_batches, event_labels = get_eventbatch_seq_eval(self.params, self.test_data)
            # pred = np.zeros(test_batches_num)
            # for k in range(test_batches_num):
            #     batch_data = test_batches[k]
            #     fetches = [self.ops['predict'], self.ops['embedding_eval'], self.ops['cellstates_eval']]
            #     batch_feed_dict = self.get_batch_feed_dict_test(batch_data, node_embedding, node_cellstates)
            #     self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
            #     predict, node_embedding, node_cellstates = self.sess.run(fetches, feed_dict=batch_feed_dict)
            #     top1index = np.argmax(predict)
            #     pred[k] = top1index
            # precision = metrics.precision_score(event_labels, pred, average=None)
            # recall = metrics.recall_score(event_labels, pred, average=None)
            # f1 = metrics.f1_score(event_labels, pred, average=None)
            # print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)



    # def eval_test(self):
    #     print('start test')
    #     cut_off = 20
    #     evalutation_point_count = 0
    #     mrr, recall, ndcg20, ndcg = 0.0, 0.0, 0.0, 0.0
    #     # precision, recall, f1, recall20 = 0.0, 0.0, 0.0, 0.0
    #     test_batches_num, test_batches = get_batch_seq_eval(self.params, self.test_data)
    #     node_embedding = [None for node_type in range(self.params['node_type_numbers'])]
    #     node_cellstates = [None for node_type in range(self.params['node_type_numbers'])]
    #     for node_type in range(self.params['node_type_numbers']):
    #         node_embedding[node_type] = self._node_embedding[node_type].eval(session=self.sess)
    #         node_cellstates[node_type] = self._node_cellstates[node_type].eval(session=self.sess)
    #     # weight = self.weights['edge_weights'][0][0].eval(session=self.sess)
    #     # print(weight)
    #     for k in range(test_batches_num):
    #         batch_data = test_batches[k]
    #         fetches = [self.ops['predict'], self.ops['embedding_eval'], self.ops['cellstates_eval']]
    #         batch_feed_dict = self.get_batch_feed_dict_eval(batch_data, node_embedding, node_cellstates)
    #         self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
    #         predict, node_embedding, node_cellstates = self.sess.run(fetches, feed_dict=batch_feed_dict)
    #         out_idx = batch_data[1]
    #         # topindex = sorted(range(len(predict)), key=lambda x: predict[x], reverse=True)[:20]
    #         # predict[:] = 0
    #         # for index in topindex:
    #         #     predict[index] = 1
    #         # precision += me
    #         ranks = np.zeros_like(out_idx)
    #         evalutation_point_count = len(ranks)
    #         # print(predict>predict[out_idx[0]])
    #         for i in range(len(ranks)):
    #             ranks[i] = (predict>predict[out_idx[i]]).sum(axis=0)+1
    #         print(np.mean(ranks))
    #         time.sleep(1)
    #         rank_ok = ranks < cut_off
    #         recall += rank_ok.sum()/evalutation_point_count
    #         mrr += (1.0 / ranks[rank_ok]).sum()/evalutation_point_count
    #         ndcg20 += (1.0 / np.log2(1.0 + ranks[rank_ok])).sum()/evalutation_point_count
    #         ndcg += (1.0 / np.log2(1.0 + ranks)).sum()/evalutation_point_count
    #         # print(rank_ok.sum()/evalutation_point_count, (1.0 / ranks[rank_ok]).sum()/evalutation_point_count,
    #         #       (1.0 / np.log2(1.0 + ranks[rank_ok])).sum()/evalutation_point_count,
    #         #       (1.0 / np.log2(1.0 + ranks)).sum()/evalutation_point_count)
    #     print(recall / test_batches_num, mrr / test_batches_num, ndcg20 / test_batches_num, ndcg / test_batches_num)







