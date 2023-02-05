from typing import Tuple, List, Any, Sequence
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import sys, os
import json
from utils.data_manager import *
from models.hegraph import HeGraph
import time

class BasicModel(object):
    @classmethod
    def default_params(cls):
        return{
            'num_epochs': 20,
            'hidden_size': 128,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 5.0,
            'eval_point': 1e1,
            'decay_step': 4e1,
            'decay': 0.98,
            'patience': 5,
            'node_type_numbers': 3,
            'batch_event_numbers': 200,
            # 'batch_event_numbers_test': 200,
            # 'use_propagation_attention': False,
            'graph_rnn_cell': 'lstm',
            'graph_rnn_activation': 'tanh',
            'keep_prob': 1.0,
            'use_different_cell': False
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
        print(self.params['embedding_out'])

        self.log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob'])
        self.checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob'])
        self.log_file = os.path.join(log_dir, self.log_file)
        self.checkpoint_dir = os.path.join(save_dir, self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        save_event_data(data_dir, self.params['batch_event_numbers'])
        self.train_data = load_event_data(os.path.join(data_dir, 'event_sorted_small_train.txt'))
        self.valid_data = load_event_data(os.path.join(data_dir, 'event_sorted_small_valid.txt'))
        self.test_data = load_event_data(os.path.join(data_dir, 'event_sorted_small_test.txt'))
        print('load data done')
        # self.train_data, self.valid_data, self.test_data = load_event_data(os.path.join(data_dir, 'event_sorted.txt'), self.params['batch_event_numbers'])
        self.hegraph = HeGraph(self.params, self.train_data, self.valid_data, self.test_data)
        self.hegraph.build_hegraph()
        self.params['n_nodes_pertype'] = self.hegraph.get_totalnum_pertype()
        print('hegraph built')
        print(self.params['n_nodes_pertype'])
        self.train_data = BatchData(self.params, self.train_data)
        self.valid_data = BatchData(self.params, self.valid_data)
        self.test_data = BatchData(self.params, self.test_data)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            # self.make_evalmodel()
            self.make_train_step()
            # Restore/initialize variables:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            init_from = args.get('--init_from')
            if init_from is not None:
                ckpt = tf.train.get_checkpoint_state(init_from)
                print('ckpt')
                print(ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('init from:'+init_from)
            else:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                print('Randomly initialize model')

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def build_specific_graph_model(self, state) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_model(self):
        #compute loss
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                # self.ops['loss'], self.ops['embedding'], self.ops['celstates'], self.ops['predict'], self.scores, self.negscores, self.shape = self.build_specific_graph_model()
                self.ops['loss'], self.ops['embedding'] = self.build_specific_graph_model()
                # self.ops['loss_eval'], self.ops['embedding_eval'], self.ops['celstates_eval'], self.ops['predict_eval'] = self.build_specific_graph_model('eval')
            else:
                self.ops['loss'], self.ops['embedding'], self.ops['predict'] = self.build_specific_graph_model()
                # self.ops['loss_eval'], self.ops['embedding_eval'], self.ops['predict_eval'] = self.build_specific_graph_model('eval')
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    # def make_evalmodel(self):
    #     self.ops['predict'], self.ops['embedding_eval'], self.ops['cellstates_eval'] = self.compute_final_node_representations_peratch_eval()
    #     print('evalmodel build')

    def make_train_step(self):
        # trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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
        # grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'])
        print(tf.trainable_variables())
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

    # def gen_sampling_table_pertype(self, tabel_z, tabel_T, batch_data):
    #     tabel_size = self.params['table_size']
    #     power = self.params['neg_power']
    #     nodes_degree_last, nodes_degree_cur = self.hegraph.get_curdegree_pertype(batch_data)
    #     for event in range(self.params['batch_event_numbers']):
    #         for type in range(self.params['node_type_numbers']):
    #             for node in batch_data[event][type]:
    #                 # print('cur%i\tlast%i'%(nodes_degree_cur[type][node], nodes_degree_last[type][node]))
    #                 if node not in nodes_degree_last[type]:
    #                     last_feq = 0
    #                 else:
    #                     last_feq = math.pow(nodes_degree_last[type][node], power)
    #                 tabel_F = math.pow(nodes_degree_cur[type][node], power) - last_feq
    #
    #                 tabel_z[type] += tabel_F
    #                 if len(tabel_T[type]) < tabel_size:
    #                     substituteNum = tabel_F
    #                 else:
    #                     substituteNum = tabel_F*tabel_size/tabel_z[type]
    #
    #                 ret = random.random()
    #                 if ret < tabel_F - math.floor(tabel_F):
    #                     substituteNum = int(substituteNum) + 1
    #                 else:
    #                     substituteNum = int(substituteNum)
    #                 for _ in range(substituteNum):
    #                     if len(tabel_T[type]) < tabel_size:
    #                         tabel_T[type].append(node)
    #                     else:
    #                         substitute = random.randint(0, len(tabel_T))
    #                         tabel_T[substitute] = node
    #
    #     return tabel_z, tabel_T

    # def gen_negative_batchdata(self, batch_data, tabel_T):
    #     batch_data_neg = [[[]for _ in range(self.params['node_type_numbers'])]
    #                         for _ in range(self.params['batch_event_numbers'])]
    #
    #     for event_neg in range(self.params['batch_event_numbers']):
    #         for type in range(self.params['node_type_numbers']):
    #             tabel_size = len(tabel_T[type])
    #             while(len(batch_data_neg[event_neg][type])<self.params['negative_ratio']):
    #                 neg_node = tabel_T[type][random.randint(0, tabel_size - 1)]
    #                 if (neg_node in batch_data[event_neg][type]) or (neg_node in batch_data_neg[event_neg][type]):
    #                     continue
    #                 batch_data_neg[event_neg][type].append(neg_node)

                # for neg_radio in range(self.params['negative_ratio']):
                #     while(len(batch_data_neg[event_neg][type][neg_radio]) < len(batch_data[event_neg][type])):
                #         neg_node = tabel_T[type][random.randint(0, tabel_size-1)]
                #         if neg_node in batch_data[event_neg][type] or neg_node in batch_data_neg[event_neg][type][neg_radio]:
                #             continue
                #         batch_data_neg[event_neg][type][neg_radio].append(neg_node)

        # return batch_data_neg

    # def get_batch_feed_dict(self, batch_data, batch_data_neg):
    #     batch_feed_dict = {}
    #     for event in range(self.params['batch_event_numbers']):
    #         for node_type in range(self.params['node_type_numbers']):
    #             event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
    #             batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(batch_data[event][node_type], dtype=np.int32)
    #             event_partition[batch_data[event][node_type]] = 1
    #             batch_feed_dict[self.placeholders['event_partition_idx_ph'][event][node_type]] = event_partition
    #             batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][0]] = np.where(event_partition==0)[0].tolist()
    #             batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][1]] = np.where(event_partition==1)[0].tolist()
    #             batch_feed_dict[self.placeholders['events_negnodes_type_ph'][event][node_type]] = np.asarray(batch_data_neg[event][node_type], dtype=np.int32)
    #     return batch_feed_dict

    # def get_batch_feed_dict_eval(self, batch_data, node_embedding, node_cellstates):
    #     batch_feed_dict = {}
    #     for node_type in range(self.params['node_type_numbers']):
    #         event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
    #         batch_feed_dict[self.placeholders['events_nodes_type_eval_ph'][node_type]] = np.asarray(
    #             batch_data[node_type], dtype=np.int32)
    #         event_partition[batch_data[node_type]] = 1
    #         batch_feed_dict[self.placeholders['event_partition_idx_eval_ph'][node_type]] = event_partition
    #         batch_feed_dict[self.placeholders['event_stitch_idx_eval_ph'][node_type][0]] = \
    #         np.where(event_partition == 0)[0].tolist()
    #         batch_feed_dict[self.placeholders['event_stitch_idx_eval_ph'][node_type][1]] = \
    #         np.where(event_partition == 1)[0].tolist()
    #
    #     # for node_type in range(self.params['node_type_numbers']):
    #         batch_feed_dict[self.placeholders['node_embedding_eval_ph'][node_type]] = node_embedding[node_type]
    #         batch_feed_dict[self.placeholders['node_cellstates_eval_ph'][node_type]] = node_cellstates[node_type]
    #     return batch_feed_dict


    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        print('batches'+str(train_batches_num))
        self.node_embedding_cur = [None for _ in range(self.params['node_type_numbers'])]
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            self.node_cellstates_cur = [None for _ in range(self.params['node_type_numbers'])]
        # self.tabel_z = [0 for _ in range(self.params['node_type_numbers'])]
        # self.tabel_T = [[] for _ in range(self.params['node_type_numbers'])]
        # total_time_start = time.time()
        # self.eval_test() ###################################
        # self.embedding_init = [None for _ in range(self.params['node_type_numbers'])]
        # self.cellstates_init = [None for _ in range(self.params['node_type_numbers'])]
        # for node_type in range(self.params['node_type_numbers']):
        #     self.embedding_init[node_type] = self._node_embedding[node_type].eval(session=self.sess)
        #     if self.params['graph_rnn_cell'].lower() == 'lstm':
        #         self.cellstates_init[node_type] = self._node_cellstates[node_type].eval(session=self.sess)

        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []

                # num_batches, batches, lebels, current = get_batch_seq(self.params, self.train_data, current)
                epoch_flag = False
                print('start epoch %i'%(epoch))


                # for node_type in range(self.params['node_type_numbers']):
                #     self.node_embedding_cur[node_type] = self.embedding_init[node_type]
                #     if self.params['graph_rnn_cell'].lower() == 'lstm':
                #         self.node_cellstates_cur[node_type] = self.cellstates_init[node_type]
                #
                # batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train')
                # em_batch_feed_dict = self.get_embatch_feed_dict('train')
                # self.sess.run(self._assign_ops, feed_dict=em_batch_feed_dict)
                #
                # if self.params['graph_rnn_cell'].lower() == 'lstm':
                #     fetches = [self.ops['loss'], self.ops['embedding'], self.ops['celstates'],
                #                self.ops['global_step'], self.ops['lr'], self.ops['predict'],self.scores, self.negscores, self.ops['train_op']]
                #     cost, self.node_embedding_cur, self.node_cellstates_cur, step, lr, predict,scores, negscores, _ = \
                #         self.sess.run(fetches, feed_dict=batch_feed_dict)
                #     print('###')
                #     # print(scores)
                #     # print(negscores)
                #     print('###')
                # else:
                #     fetches = [self.ops['loss'], self.ops['embedding'], self.ops['global_step'],
                #                self.ops['lr'], self.ops['predict'], self.ops['train_op']]
                #     cost, self.node_embedding_cur, step, lr, predict, scores, negscores, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)
                #
                # for node_type in range(self.params['node_type_numbers']):
                #     self.embedding_init[node_type] = self._node_embedding[node_type].eval(session=self.sess)
                #     # self.node_embedding_cur[node_type] = np.repeat(embedding_init, self.params['n_nodes_pertype'])
                #     if self.params['graph_rnn_cell'].lower() == 'lstm':
                #         self.cellstates_init[node_type] = self._node_cellstates[node_type].eval(session=self.sess)
                #         # self.node_cellstates_cur[node_type] = np.repeat(cellstates_init, self.params['n_nodes_pertype'])
                #
                # print('Step {}\tlr: {:.6f}\tcost:{:.6f}'.format(step, lr, cost))
                #
                # if step == 1:
                #     valid_loss = self.validation()
                #     print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))

                while not epoch_flag:
                # for k in range(num_batches):
                    # batch_data = batches[k]
                    # batch_data, batch_label, epoch_flag = self.train_data.next_batch()
                    # self.tabel_z, self.tabel_T = self.gen_sampling_table_pertype(self.tabel_z, self.tabel_T, batch_data)
                    # batch_data_neg = self.gen_negative_batchdata(batch_data, self.tabel_T)
                    # em_batch_feed_dict = self.get_embatch_feed_dict('train')
                    # self.sess.run(self._assign_ops, feed_dict=em_batch_feed_dict)
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train')
                    if self.params['graph_rnn_cell'].lower() == 'lstm':
                        # fetches = [self.ops['loss'], self.ops['embedding'], self.ops['celstates'],
                        #            self.ops['global_step'], self.ops['lr'], self.ops['predict'], self.scores, self.negscores,self.shape, self.ops['train_op']]
                        fetches = [self.ops['loss'], self.ops['embedding'], self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                        # cost, self.node_embedding_cur, self.node_cellstates_cur, step, lr, predict, scores, negscores,shape, _ = \
                        #     self.sess.run(fetches, feed_dict=batch_feed_dict)
                        cost,embedding, step, lr, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)
                        if step == 1:
                            valid_loss = self.validation()
                            print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        # print('__________________')
                        # print(np.mean(scores))
                        # print(np.mean(negscores, axis=0))
                        # print(scores)
                        # print(negscores)
                        # print('--------')
                        # print(shape)
                        # print(rmse)
                        # print(negrmse)
                    else:
                        fetches = [self.ops['loss'], self.ops['embedding'], self.ops['global_step'],
                                   self.ops['lr'], self.ops['predict'], self.ops['train_op']]
                        cost, self.node_embedding_cur, step, lr, predict, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)

                    # for event in range(self.params['batch_event_numbers']):
                    #     out_idx = batch_data[event][1]
                    #     ranks = np.zeros_like(out_idx)
                    #     for i in range(len(ranks)):
                    #         ranks[i] = (predict[event] > predict[event][out_idx[i]]).sum(axis=0) + 1
                    #     avg_rank = np.mean(ranks)
                    #     print('avv')
                    #     print(avg_rank)
                    #     time.sleep(1)

                    #     printrank += avg_rank
                    # print('avgrank')
                    # print(printrank/self.params['batch_event_numbers'])
                    # print('')
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
                        # print(k)
                        return
                    if step % (self.params['decay_step']/40) == 0:
                        avgc = np.mean(epoch_loss)
                        log_out.write('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        print(cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        sys.stdout.flush()
                # if step == 1 or step % self.params['eval_point'] == 0:
                print('start valid')
                valid_loss = self.validation()
                log_out.write('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                # self.test()
                if valid_loss < best_loss:
                    best_epoch = epoch
                    best_loss = valid_loss
                    ckpt_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
                    self.saver.save(self.sess, ckpt_path, global_step=step)
                    log_out.write('model saved to {}'.format(ckpt_path))
                    print('model saved to {}'.format(ckpt_path))
                    sys.stdout.flush()
                # epoch_time = time.time() - total_time_start
                if epoch-best_epoch >= self.params['patience']:
                    log_out.write('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                    print('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                    break
            log_out.write('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            print('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            save_trained_embeddings(embedding, self.params['embedding_out'])
            print('save embeddings')
            print('start test')
            # self.test()

    def validation(self):
        valid_batches_num = self.valid_data.get_batch_num()
        # valid_batches_num, valid_batches = get_batch_seq(self.params, self.valid_data)
        valid_loss = []
        epoch_flag = False
        # self.node_embedding_init = [None for _ in range(self.params['node_type_numbers'])]
        # self.node_embedding_cur = [None for _ in range(self.params['node_type_numbers'])]
        # if self.params['graph_rnn_cell'].lower() == 'lstm':
        #     self.node_cellstates_init = [None for _ in range(self.params['node_type_numbers'])]
        #     self.node_cellstates_cur = [None for _ in range(self.params['node_type_numbers'])]
        # for node_type in range(self.params['node_type_numbers']):
        #     self.node_embedding_init[node_type] = self._node_embedding[node_type].eval(session=self.sess)
        #     self.node_embedding_cur[node_type] = self.node_embedding_init[node_type]
        #     if self.params['graph_rnn_cell'].lower() == 'lstm':
        #         self.node_cellstates_init[node_type] = self._node_cellstates[node_type].eval(session=self.sess)
        #         self.node_cellstates_cur[node_type] = self.node_cellstates_init[node_type]
        print(valid_batches_num)
        # self.node_embedding_validcur = [None for _ in range(self.params['node_type_numbers'])]
        # self.node_cellstates_validcur = [None for _ in range(self.params['node_type_numbers'])]
        # for node_type in range(self.params['node_type_numbers']):
        #     self.node_embedding_validcur[node_type] = self.node_embedding_cur[node_type]
        #     if self.params['graph_rnn_cell'].lower() == 'lstm':
        #         self.node_cellstates_validcur[node_type] = self.node_cellstates_cur[node_type]
        while not epoch_flag:
            # batch_data, batch_label,  = self.valid_data.next_batch()
            # batch_data = valid_batches[k]
            # em_batch_feed_dict = self.get_embatch_feed_dict('valid')
            # self.sess.run(self._assign_ops, feed_dict=em_batch_feed_dict)
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                # fetches = [self.ops['embedding'], self.ops['celstates'], self.ops['loss'], self.scores, self.negscores]
                fetches = [self.ops['loss']]
                # self.tabel_z, self.tabel_T = self.gen_sampling_table_pertype(self.tabel_z, self.tabel_T, batch_data)
                # batch_data_neg = self.gen_negative_batchdata(batch_data, self.tabel_T)
                batch_feed_dict, epoch_flag = self.get_batch_feed_dict('valid')
                # em_batch_feed_dict = self.get_embatch_feed_dict('valid')
                # self.sess.run(self._assign_ops, feed_dict=em_batch_feed_dict)
                # for node_type in range(self.params['node_type_numbers']):
                #     batch_feed_dict[self._node_embedding[node_type]] = node_embedding[node_type]
                #     batch_feed_dict[self._node_cellstates[node_type]] = node_cellstates[node_type]
                # self.node_embedding_validcur, self.node_cellstates_validcur, cost, scores, negscores = self.sess.run(fetches, feed_dict=batch_feed_dict)
                cost = self.sess.run(fetches, feed_dict=batch_feed_dict)
                # print('__________________')
                # print(np.mean(scores))
                # print(np.mean(negscores,axis=0))
                # print('--------')
                # print(rmse)
                # print(negrmse)
            else:
                fetches = [self.ops['embedding'], self.ops['loss']]
                batch_feed_dict, epoch_flag = self.get_batch_feed_dict('valid')
                self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
                self.node_embedding_cur, cost = self.sess.run(fetches, feed_dict=batch_feed_dict)
            if np.isnan(cost):
                print('Evaluation loss Nan!')
                # print(k)
                sys.exit(1)
            valid_loss.append(cost)
        # for node_type in range(self.params['node_type_numbers']):
        #     batch_feed_dict[self.placeholders['node_embedding_ph'][node_type]] = self.node_embedding_init[node_type]
        #     if self.params['graph_rnn_cell'].lower() == 'lstm':
        #         batch_feed_dict[self.placeholders['node_cellstates_ph'][node_type]] = self.node_cellstates_init[node_type]
        # self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
        return np.mean(valid_loss)


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







