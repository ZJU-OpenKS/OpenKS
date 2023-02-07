import math
import random

import numpy as np
import tensorflow as tf

from models.basic_model import BasicModel
from utils.data_manager import BatchData
from utils.inits import glorot_init, zeros_init


class GnnLinkModel(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'negative_ratio': 5,
            'table_size': 1e8,
            'neg_power': 0.75,
            'aggregator_type': 'attention'
        })
        return params

    def _create_placeholders(self):
            self.placeholders['events_nodes_type_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                        name='node_type%i_event%i_ph'%(node_type, event))
                                                        for node_type in range(self._type_num)]
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['events_negnodes_type_ph'] =[[tf.placeholder(tf.int32, shape=[None],
                                                name='negnode_type%i_event%i_ph'%(node_type, event))
                                                for node_type in range(self._type_num)]
                                                for event in range(self._eventnum_batch)]

            self.placeholders['event_partition_idx_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                        name='event_partition_idx_type%i_event%i_ph'%(node_type, event))
                                                        for node_type in range(self._type_num)]
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['event_stitch_idx_ph'] = [[[
                tf.placeholder(tf.int32, shape=[None]),
                tf.placeholder(tf.int32, shape=[None])
            ] for node_type in range(self._type_num)] for event in range(self._eventnum_batch)]

            self.placeholders['node_embedding_eval_ph'] = [tf.placeholder(tf.float64,
                                                        shape=[self._num_node_type[node_type], self._h_dim],
                                                        name='node_type%i_embedding_ph' % node_type)
                                                        for node_type in range(self._type_num)]

            self.placeholders['node_cellstates_eval_ph'] = [tf.placeholder(tf.float64,
                                                        shape=[self._num_node_type[node_type], self._h_dim],
                                                        name='node_type%i_cellstates_ph' % node_type)
                                                        for node_type in range(self._type_num)]

    def _create_variables(self):
        cur_seed = random.getrandbits(32)
        self._node_embedding = [tf.get_variable('node_type%i_embedding'%(node_type),
                                shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]
        # self._node_cellstates = [tf.get_variable('node_type%i_cellstates'%(node_type),
        #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
        #                         initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        #                         for node_type in range(self._type_num)]
        self._node_cellstates = [tf.get_variable('node_type%i_cellstates'%(node_type),
                                shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=False,
                                initializer=tf.zeros_initializer())
                                for node_type in range(self._type_num)]

        self._assign_ops = []
        for node_type in range(self._type_num):
            self._assign_ops += [
                tf.assign(self._node_embedding[node_type], self.placeholders['node_embedding_eval_ph'][node_type])]
            self._assign_ops += [
                tf.assign(self._node_cellstates[node_type], self.placeholders['node_cellstates_eval_ph'][node_type])]
        self._assign_ops = tf.group(*self._assign_ops)

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._negative_ratio = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']

        self._create_placeholders()
        self._create_variables()

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['type_weights'] = [tf.Variable(glorot_init([self._h_dim, self._h_dim],
                                        name='gnn_type_weights_typ%i' % node_type))
                                        for node_type in range(self._type_num)]
        if self.params['use_type_bias']:
            self.weights['type_biases'] = [tf.Variable(zeros_init([self._num_node_type[node_type]],
                                            name='gnn_edge_biases_typ%i' % node_type))
                                           for node_type in range(self._type_num)]
            # self.weights['edge_biases'] = [tf.Variable(zeros_init([self._h_dim],
            #                             name='gnn_edge_biases_typ%i' % node_type))
            #                             for node_type in range(self._type_num)]

        # aggregator_type = self.params['aggregator_type'].lower()
        # if aggregator_type == 'mean':
        #     aggregator = MeanAggregator
        # elif aggregator_type == 'attention':
        #     aggregator = AttentionAggregator
        # else:
        #     raise Exception('Unknown aggregator: ', aggregator_type)
        #
        # self.weights['aggregators'] = aggregator()

        cell_type = self.params['graph_rnn_cell'].lower()
        if cell_type == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell(self._h_dim, activation=activation_fun)
        elif cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(self._h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(self._h_dim, activation=activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.params['graph_state_keep_prob'])

        self.weights['rnn_cells'] = cell

        self.tabel_z = [0 for _ in range(self._type_num)]
        self.tabel_T = [[] for _ in range(self._type_num)]

    def compute_final_node_representations_perbatch(self):
        loss_pertype = [[None for node_type in range(self._type_num)] for event_id in range(self._eventnum_batch)]
        node_vec = [[None]*self._type_num]*(self._eventnum_batch+1)
        node_vec[0] = self._node_embedding
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            node_cel = [[None]*self._type_num]*(self._eventnum_batch+1)
            node_cel[0] = self._node_cellstates
        predict = [[] for event_id in range(self._eventnum_batch)]
        for event_id in range(self._eventnum_batch):
            event_states = tf.zeros([self._h_dim], dtype=tf.float64)
            for node_type in range(self._type_num):
                states = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                self.placeholders['events_nodes_type_ph'][event_id][node_type])
                states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
                event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])

            for node_type in range(self._type_num):
                node_vec_part = tf.dynamic_partition(node_vec[event_id][node_type],
                                                    self.placeholders['event_partition_idx_ph'][event_id][node_type],2)
                target_embedding = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                    self.placeholders['events_nodes_type_ph'][event_id][node_type])
                neg_target_embedding = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                    self.placeholders['events_negnodes_type_ph'][event_id][node_type])
                if self.params['graph_rnn_cell'].lower() == 'lstm':
                    node_cel_part = tf.dynamic_partition(node_cel[event_id][node_type],
                                                     self.placeholders['event_partition_idx_ph'][event_id][node_type],2)
                    target_celstates = tf.nn.embedding_lookup(node_cel[event_id][node_type],
                                                    self.placeholders['events_nodes_type_ph'][event_id][node_type])
                # sent_states_aggregated = tf.zeros_like(target_embedding)
                context_states = tf.zeros_like(target_embedding)
                neg_context_states = tf.zeros_like(neg_target_embedding)
                sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
                predict_states = tf.zeros_like(node_vec[event_id][node_type])

                for other_type in [type for type in range(self._type_num) if type != node_type]:
                    other_states = tf.nn.embedding_lookup(node_vec[event_id][other_type],
                                                    self.placeholders['events_nodes_type_ph'][event_id][other_type])
                    other_states_aggregated = self.weights['aggregators']((target_embedding, other_states))
                    neg_other_states_aggregated = self.weights['aggregators']((neg_target_embedding, other_states))
                    other_states_predicted = self.weights['aggregators']((node_vec[event_id][node_type], other_states))
                    predict_states += tf.matmul(other_states_predicted, self.weights['type_weights'][other_type])
                    context_states += tf.matmul(other_states_aggregated, self.weights['type_weights'][other_type])
                    neg_context_states += tf.matmul(neg_other_states_aggregated, self.weights['type_weights'][other_type])
                    # other_contextstates_aggregated = MeanAggregator((target_embedding, other_states))
                    # context_states += tf.matmul(other_contextstates_aggregated, self.weights['edge_weights'][other_type])
                if node_type == 2:
                    context_states_predict = predict_states
                    target_embedding_predict = node_vec[event_id][node_type]
                    score = tf.reduce_sum(tf.multiply(context_states_predict, target_embedding_predict), 1) + self.weights['edge_biases'][node_type]
                    predict[event_id] = tf.nn.sigmoid(score)

                if self.params['graph_rnn_cell'].lower() == 'lstm':
                    _, new_target_states_ch = self.weights['rnn_cells'](inputs=sent_states,
                                                                        state=(target_celstates, target_embedding))
                    new_target_celstates = new_target_states_ch[0]
                    new_target_embedding = new_target_states_ch[1]
                    node_cel[event_id + 1][node_type] = tf.dynamic_stitch(
                        self.placeholders['event_stitch_idx_ph'][event_id][node_type],
                        [node_cel_part[0], new_target_celstates])
                else:
                    new_target_embedding = self.weights['rnn_cells'](inputs=sent_states, state=target_embedding)[1]

                node_vec[event_id + 1][node_type] = tf.dynamic_stitch(
                    self.placeholders['event_stitch_idx_ph'][event_id][node_type],
                    [node_vec_part[0], new_target_embedding])


                # context_states = sent_states_aggregated

                # loss_pertype[event_id][node_type] = -tf.reduce_mean(
                #     tf.log_sigmoid(tf.reduce_sum(tf.multiply(target_embedding, context_states), axis=1)))
                true_b = tf.nn.embedding_lookup(self.weights['type_biases'][node_type],
                                                self.placeholders['events_nodes_type_ph'][event_id][node_type])
                true_logits = tf.reduce_sum(tf.multiply(context_states, target_embedding), axis=1) + true_b


                neg_b = tf.nn.embedding_lookup(self.weights['type_biases'][node_type],
                                               self.placeholders['events_negnodes_type_ph'][event_id][node_type])
                neg_b_vec = tf.reshape(neg_b, [self.params['negative_ratio']])
                neg_logits = tf.reduce_sum(tf.multiply(neg_context_states, neg_target_embedding), axis=1) + neg_b_vec
                # neg_logits = tf.matmul(neg_context_states, neg_target_embedding, transpose_b=True) + neg_b_vec

                # for neg_radio in range(self._negative_ratio):
                #     neg_target_states = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                #                         self.placeholders['events_negnodes_type_ph'][event_id][node_type][neg_radio])
                #
                #     loss_pertype[event_id][node_type] += -tf.reduce_mean(tf.log_sigmoid(
                #         -1 * tf.reduce_sum(tf.multiply(neg_target_states, context_states), axis=1)))

                true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(true_logits), logits=true_logits)
                neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(neg_logits), logits=neg_logits)
                target_types_num = tf.cast(tf.shape(self.placeholders['events_nodes_type_ph'][event_id][node_type])[0],
                                           tf.float64)
                loss_pertype[event_id][node_type] = tf.divide(tf.reduce_sum(true_xent) + tf.reduce_sum(neg_xent),
                                                              target_types_num)

        loss = tf.reduce_mean(loss_pertype)

        return loss, node_vec[event_id+1], node_cel[event_id+1], predict

    def test(self):
        self.test_data = BatchData(self.params, self.test_data)
        test_batches_num = self.test_data.get_batch_num()
        epoch_flag = False
        cut_off = 20
        evalutation_point_count = 0
        mrr, recall, ndcg20, ndcg = 0.0, 0.0, 0.0, 0.0
        # precision, recall, f1, recall20 = 0.0, 0.0, 0.0, 0.0
        while not epoch_flag:
            fetches = [self.ops['embedding'], self.ops['celstates'], self.ops['predict']]
            batch_feed_dict, batch_label, epoch_flag = self.get_batch_feed_dict_test()
            self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
            self.node_embedding_cur, self.node_cellstates_cur, predict = self.sess.run(fetches, feed_dict=batch_feed_dict)
            out_idx = batch_label
            ranks = np.zeros_like(out_idx)
            evalutation_point_count += len(ranks)
            for i in range(len(ranks)):
                ranks[i] = (predict>predict[out_idx[i]]).sum(axis=0)+1
            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0/ranks[rank_ok]).sum()
            ndcg20 += (1.0/np.log2(1.0+ranks[rank_ok])).sum()
            ndcg += (1.0/np.log2(1.0+ranks)).sum()
        print(recall/evalutation_point_count, recall/evalutation_point_count, mrr/evalutation_point_count, ndcg/evalutation_point_count)

    def gen_sampling_table_pertype(self, tabel_z, tabel_T, batch_data):
        tabel_size = self.params['table_size']
        power = self.params['neg_power']
        nodes_degree_last, nodes_degree_cur = self.hegraph.get_curdegree_pertype(batch_data)
        for event in range(self._eventnum_batch):
            for type in range(self._type_num):
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
        batch_data_neg = [[[]for _ in range(self._type_num)]
                            for _ in range(self._eventnum_batch)]
        for event_neg in range(self._eventnum_batch):
            for type in range(self._type_num):
                tabel_size = len(tabel_T[type])
                while(len(batch_data_neg[event_neg][type])<self.params['negative_ratio']):
                    neg_node = tabel_T[type][random.randint(0, tabel_size - 1)]
                    if (neg_node in batch_data[event_neg][type]) or (neg_node in batch_data_neg[event_neg][type]):
                        continue
                    batch_data_neg[event_neg][type].append(neg_node)
        return batch_data_neg

    def get_batch_feed_dict(self, state):
        batch_feed_dict = {}
        if state == 'train':
            batch_data, batch_label, epoch_flag = self.train_data.next_batch()
        elif state == 'valid':
            batch_data, batch_label, epoch_flag = self.valid_data.next_batch()
        else:
            print('state wrong')
        self.tabel_z, self.tabel_T = self.gen_sampling_table_pertype(self.tabel_z, self.tabel_T, batch_data)
        batch_data_neg = self.gen_negative_batchdata(batch_data, self.tabel_T)
        for event in range(self.params['batch_event_numbers']):
            for node_type in range(self.params['node_type_numbers']):
                event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(
                    batch_data[event][node_type], dtype=np.int32)
                event_partition[batch_data[event][node_type]] = 1
                batch_feed_dict[self.placeholders['event_partition_idx_ph'][event][node_type]] = event_partition
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][0]] = \
                np.where(event_partition == 0)[0].tolist()
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][1]] = \
                np.where(event_partition == 1)[0].tolist()
                batch_feed_dict[self.placeholders['events_negnodes_type_ph'][event][node_type]] = \
                    np.asarray(batch_data_neg[event][node_type], dtype=np.int32)
        return batch_feed_dict, epoch_flag

    def get_batch_feed_dict_test(self):
        batch_feed_dict = {}
        batch_data, batch_label, epoch_flag = self.test_data.next_batch()
        for node_type in range(self.params['node_type_numbers']):
            batch_feed_dict[self.placeholders['node_embedding_eval_ph'][node_type]] = self.node_embedding_cur[
                node_type]
            batch_feed_dict[self.placeholders['node_cellstates_eval_ph'][node_type]] = self.node_cellstates_cur[
                node_type]

        for event in range(self.params['batch_event_numbers']):
            for node_type in range(self.params['node_type_numbers']):
                event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(
                    batch_data[event][node_type], dtype=np.int32)
                event_partition[batch_data[event][node_type]] = 1
                batch_feed_dict[self.placeholders['event_partition_idx_ph'][event][node_type]] = event_partition
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][0]] = \
                    np.where(event_partition == 0)[0].tolist()
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][1]] = \
                    np.where(event_partition == 1)[0].tolist()
        return batch_feed_dict, batch_label, epoch_flag








