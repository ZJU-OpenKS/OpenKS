from typing import Tuple, List, Any, Sequence
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import sys, os
import json
import time
import math
import random
from utils.data_manager import load_event_data, get_batch_seq
from models.hegraph import HeGraph

class BasicModel(object):
    @classmethod
    def default_params(cls):
        return{
            'num_epochs': 10,
            'hidden_size': 100,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'eval_point': 5000,
            'decay_step': 2000,
            'decay': 0.98,
            'patience': 5,
            'node_type_numbers': 3,
            'batch_event_numbers': 200,
            # 'use_propagation_attention': False,
            'aggregator_type': 'mean',
            'use_type_bias': True,
            'graph_rnn_cell': 'lstm',
            'graph_rnn_activation': 'tanh',
            'graph_state_keep_prob': 0.8,
            'negative_ratio': 5,
            'table_size': 1e8,
            'neg_power': 0.75
        }

    def __init__(self, args):
        self.args = args
        data_dir = args['--data_dir']
        log_dir = args.get('--log_dir')
        save_dir = args.get('--save_dir')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        params = self.default_params()
        config_file = args.get('--config_file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--conifg')
        if config is not None:
            params.update((json.loads(config)))
        self.params = params

        self.log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' + str(self.params['decay_step']) + '_k_' + str(self.params['graph_state_keep_prob'])
        self.checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' + str(self.params['decay_step']) + '_k_' + str(self.params['graph_state_keep_prob'])
        self.log_file = os.path.join(log_dir, self.log_file)
        self.checkpoint_dir = os.path.join(save_dir, self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.train_data = load_event_data(os.path.join(data_dir, 'event_train.txt'))
        self.valid_data = load_event_data(os.path.join(data_dir, 'event_valid.txt'))
        self.test_data = load_event_data(os.path.join(data_dir, 'event_test.txt'))
        print(len(self.train_data))
        self.hegraph = HeGraph(self.params, self.train_data, self.valid_data, self.test_data)
        self.hegraph.build_hegraph()
        self.params['n_nodes_pertype'] = self.hegraph.get_totalnum_pertype()
        print('hegraph built')

        print(self.params['n_nodes_pertype'])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            # Restore/initialize variables:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            init_from = args.get('--init_from')
            if init_from is not None:
                ckpt = tf.train.get_checkpoint_state(self.checkpointa_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                print('Randomly initialize model')

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations_perbatch(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_model(self):
        #compute loss
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['cur_node_embedding'], self.ops['cur_node_cellstates'], self.ops['loss'], self.ops['test'] = self.compute_final_node_representations_perbatch()
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print(trainable_vars)
        # if self.args.get('--freeze-graph-model'):
        #     graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
        #     filtered_vars = []
        #     for var in trainable_vars:
        #         if var not in graph_vars:
        #             filtered_vars.append(var)
        #         else:
        #             print("Freezing weights of variable %s." % var.name)
        #     trainable_vars = filtered_vars

        self.ops['lr'] = tf.maximum(1e-5, tf.train.exponential_decay(self.params['learning_rate'], self.ops['global_step'], self.params['decay_step'], self.params['decay'], staircase=True))
        optimizer = tf.train.AdamOptimizer(self.ops['lr'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_op'] = optimizer.apply_gradients(clipped_grads, global_step=self.ops['global_step'])
        print('make train step finished')
        # Initialize newly-introduced variables:
        # self.sess.run(tf.local_variables_initializer())

    def gen_sampling_table_pertype(self, tabel_z, tabel_T, batch_data):
        tabel_size = self.params['table_size']
        power = self.params['neg_power']
        nodes_degree_last, nodes_degree_cur = self.hegraph.get_curdegree_pertype(batch_data)
        for event in range(self.params['batch_event_numbers']):
            for type in range(self.params['node_type_numbers']):
                for node in batch_data[event][type]:
                    # print('cur%i\tlast%i'%(nodes_degree_cur[type][node], nodes_degree_last[type][node]))
                    if node not in nodes_degree_last[type]:
                        last_feq = 0
                    else:
                        last_feq = math.pow(nodes_degree_last[type][node], power)
                    tabel_F = math.pow(nodes_degree_cur[type][node], power) - last_feq

                    tabel_z[type] += tabel_F
                    if len(tabel_T[type]) < tabel_size:
                        substituteNum = tabel_F
                    else:
                        substituteNum = tabel_F*tabel_size/tabel_z[type]

                    ret = random.random()
                    if ret < tabel_F - math.floor(tabel_F):
                        substituteNum = int(substituteNum) + 1
                    else:
                        substituteNum = int(substituteNum)
                    for _ in range(substituteNum):
                        if len(tabel_T[type]) < tabel_size:
                            tabel_T[type].append(node)
                        else:
                            substitute = random.randint(0, len(tabel_T))
                            tabel_T[substitute] = node

        return tabel_z, tabel_T

    def gen_negative_batchdata(self, batch_data, tabel_T):
        batch_data_neg = [[[[]for _ in range(self.params['negative_ratio'])]
                            for _ in range(self.params['node_type_numbers'])]
                            for _ in range(self.params['batch_event_numbers'])]

        # tabel_size = len(tabel_T[type])
        # print(tabel_size)
        for event_neg in range(self.params['batch_event_numbers']):
            for type in range(self.params['node_type_numbers']):
                tabel_size = len(tabel_T[type])
                for neg_radio in range(self.params['negative_ratio']):
                    while(len(batch_data_neg[event_neg][type][neg_radio]) < len(batch_data[event_neg][type])):
                        neg_node = tabel_T[type][random.randint(0, tabel_size-1)]
                        if neg_node in batch_data[event_neg][type] or neg_node in batch_data_neg[event_neg][type][neg_radio]:
                            continue
                        batch_data_neg[event_neg][type][neg_radio].append(neg_node)

        return batch_data_neg

        # raise Exception("Models have to implement gen_sampling_table_pertype!")
    # def get_batch_feed_dict(self, batch_data, batch_data_neg, cur_node_embedding, cur_node_cellstates):
    #     batch_feed_dict = {}
    #     for event in range(self.params['batch_event_numbers']):
    #         for node_type in range(self.params['node_type_numbers']):
    #             event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
    #             batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(batch_data[event][node_type], dtype=np.int32)
    #             event_partition[batch_data[event][node_type]] = 1
    #             batch_feed_dict[self.placeholders['partition_idx_ph'][event][node_type]] = event_partition
    #             batch_feed_dict[self.placeholders['stitch_idx_ph'][event][node_type][0]] = np.where(event_partition==0)[0].tolist()
    #             batch_feed_dict[self.placeholders['stitch_idx_ph'][event][node_type][1]] = np.where(event_partition==1)[0].tolist()
    #             for neg_radio in range(self.params['negative_ratio']):
    #                 batch_feed_dict[self.placeholders['events_negnodes_type_ph'][event][node_type][neg_radio]] = np.asarray(batch_data_neg[event][node_type][neg_radio], dtype=np.int32)
    #     for node_type in range(self.params['node_type_numbers']):
    #         batch_feed_dict[self.placeholders['node_embedding_ph'][node_type]] = cur_node_embedding[node_type]
    #         batch_feed_dict[self.placeholders['node_cellstates_ph'][node_type]] = cur_node_cellstates[node_type]
    #     return batch_feed_dict


    def get_batch_feed_dict(self, batch_data, batch_data_neg, cur_node_embedding, cur_node_cellstates):
        batch_feed_dict = {}
        batch_partition = [np.zeros(self.params['n_nodes_pertype'][node_type]) for node_type in range(self.params['node_type_numbers'])]
        batch_nodes = [[] for node_type in range(self.params['node_type_numbers'])]
        # batch_parLen = [None for node_type in range(self.params['node_type_numbers'])]
        for node_type in range(self.params['node_type_numbers']):
            for event in range(self.params['batch_event_numbers']):
                batch_partition[node_type][batch_data[event][node_type]] = 1
                # print('batch data' + str(batch_data[event][node_type]))
                for neg_radio in range(self.params['negative_ratio']):
                    batch_partition[node_type][batch_data_neg[event][node_type][neg_radio]] = 1
                    # print('negbatch data' + str(batch_data_neg[event][node_type][neg_radio]))
            # batch_parLen = len(np.where(batch_partition[node_type]==1)[0].tolist())
            batch_feed_dict[self.placeholders['batch_partition_idx_ph'][node_type]] = batch_partition[node_type]
            batch_nodes[node_type] = np.where(batch_partition[node_type]==1)[0].tolist()
            batch_feed_dict[self.placeholders['batch_stitch_idx_ph'][node_type][0]] = np.where(batch_partition[node_type]==0)[0].tolist()
            batch_feed_dict[self.placeholders['batch_stitch_idx_ph'][node_type][1]] = batch_nodes[node_type]

        for event in range(self.params['batch_event_numbers']):
            for node_type in range(self.params['node_type_numbers']):
                event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
                event_partition[batch_data[event][node_type]] = 1
                event_partition = event_partition[batch_nodes[node_type]]
                batch_feed_dict[self.placeholders['event_partition_idx_ph'][event][node_type]] = event_partition
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][0]] = np.where(event_partition==0)[0].tolist()
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][1]] = np.where(event_partition==1)[0].tolist()
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.where(event_partition==1)[0].tolist()
                # print(np.where(event_partition==1)[0].tolist())
                for neg_radio in range(self.params['negative_ratio']):
                    neg_event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
                    neg_event_partition[batch_data_neg[event][node_type][neg_radio]] = 1
                    neg_event_partition = neg_event_partition[batch_nodes[node_type]]
                    batch_feed_dict[self.placeholders['events_negnodes_type_ph'][event][node_type][neg_radio]] = np.where(neg_event_partition==1)[0].tolist()
                    # print(np.where(neg_event_partition==1)[0].tolist())
                # time.sleep(10)
        # for node_type in range(self.params['node_type_numbers']):
        #     batch_feed_dict[self.placeholders['node_embedding_ph'][node_type]] = cur_node_embedding[node_type]
        #     batch_feed_dict[self.placeholders['node_cellstates_ph'][node_type]] = cur_node_cellstates[node_type]
        return batch_feed_dict

    def train(self):
        valid_losses = []
        best_loss = 100.0
        best_epoch = -1
        best_step = -1
        self.tabel_z = [0 for _ in range(self.params['node_type_numbers'])]
        self.tabel_T = [[] for _ in range(self.params['node_type_numbers'])]
        total_time_start = time.time()
        self.cur_node_embedding = [np.zeros([self.params['n_nodes_pertype'][node_type], self.params['hidden_size']], dtype=np.float64) for node_type in range(self.params['node_type_numbers'])]
        self.cur_node_cellstates = [np.zeros([self.params['n_nodes_pertype'][node_type], self.params['hidden_size']], dtype=np.float64) for node_type in range(self.params['node_type_numbers'])]
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []
                num_batches, batches = get_batch_seq(self.params, self.train_data)
                print('batches'+str(num_batches))
                for k in range(num_batches):
                    # batch_data = batches[k]
                    batch_data = batches[k]
                    self.tabel_z, self.tabel_T = self.gen_sampling_table_pertype(self.tabel_z, self.tabel_T, batch_data)
                    batch_data_neg = self.gen_negative_batchdata(batch_data, self.tabel_T)
                    batch_feed_dict = self.get_batch_feed_dict(batch_data, batch_data_neg, self.cur_node_embedding, self.cur_node_cellstates)

                    fetches = [self.ops['loss'], self.ops['global_step'], self.ops['lr'], self.ops['cur_node_embedding'], self.ops['cur_node_cellstates'], self.ops['test'], self.ops['train_op']]
                    cost, step, lr, self.cur_node_embedding, self.cur_node_cellstates, test, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)
                    # print(test)
                    # print('node embedding')
                    # print(self.cur_node_embedding)
                    # print('cell states')
                    # print(self.cur_node_cellstates)
                    # self.ops['cur_node_states'] = tf.stop_gradient(self.ops['cur_node_states'])
                    # self.ops['cur_node_cellstates'] = tf.stop_gradient(self.ops['cur_node_cellstates'])
                    epoch_loss.append(cost)
                    if np.isnan(cost):
                        log_out(str(epoch) + ':Nan error!')
                        print(str(epoch) + ':Nan error!')
                        print(k)
                        return
                    if step == 1 or step % (self.params['decay_step']/20) == 0:
                        avgc = np.mean(epoch_loss)
                        log_out.write('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, cost))
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, cost))
                        sys.stdout.flush()
                    if step % self.params['eval_point'] == 0:
                        print('start valid')
                        valid_loss = self.eval_validation()
                        valid_losses.append(valid_loss)
                        print('valid finished')
                        log_out.write('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        if valid_loss < best_loss:
                            best_epoch = epoch
                            best_step = step
                            best_loss = valid_loss
                            ckpt_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
                            self.saver.save(self.sess, ckpt_path, global_step=step)
                            log_out.write('model saved to {}'.format(ckpt_path))
                            print('model saved to {}'.format(ckpt_path))
                            sys.stdout.flush()
                epoch_time = time.time() - total_time_start
                if epoch-best_epoch >= self.params['patience']:
                    log_out.write('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                    print('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
        log_out.write('Best evaluation loss appears in epoch {}, step {}. Lowest loss: {:.6f}'.format(epoch, step, best_loss))
        print('Best evaluation loss appears in epoch {}, step {}. Lowest loss: {:.6f}'.format(epoch, step, best_loss))

    def eval_validation(self):
        valid_batches, batches = get_batch_seq(self.params, self.valid_data)
        valid_loss = []
        for k in range(valid_batches):
            batch_data = batches[k]
            fetches = self.ops['loss']
            self.tabel_z, self.tabel_T = self.gen_sampling_table_pertype(self.tabel_z, self.tabel_T, batch_data)
            batch_data_neg = self.gen_negative_batchdata(batch_data, self.tabel_T)
            batch_feed_dict = self.get_batch_feed_dict(batch_data, batch_data_neg, self.cur_node_embedding, self.cur_node_cellstates)
            cost = self.sess.run(fetches, feed_dict=batch_feed_dict)
            if np.isnan(cost):
                print('Evaluation loss Nan!')
                print(k)
                sys.exit(1)
            valid_loss.append(cost)
        return np.mean(valid_loss)


    def eval_test(self):
        self.params['batch_event_numbers'] = 1
        test_batches, batches = get_batch_seq(self.params, self.test_data)
        for k in range(test_batches):
            batch_data = batches[k]



