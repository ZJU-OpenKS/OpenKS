import random
import sys
import time
import random

import numpy as np
import scipy.sparse as sp
import sklearn.metrics as metrics

from layers.aggregators import *
from layers.layers import *
from layers.aggregators import *
from models.basic_model import BasicModel
from utils.data_manager import *


# from scipy.sparse import linalg

class GnnEventModel_withattention_sum(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'negative_ratio': 10,
            'table_size': 1e8,
            'neg_power': 0.75,
            # 'use_event_bias': True
        })
        return params

    def get_log_file(self, params):
        log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'with_attention'
        return log_file

    def get_checkpoint_dir(self, params):
        checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'with_attention'
        return checkpoint_dir

    def make_model(self):
        #compute loss
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['loss'], self.ops['node_vec'], self.ops['node_cel'], self.ops['pred'], self.ops['neg_pred'] = self.build_specific_graph_model()
            self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict'], self.ops['node_vec_eval'], \
                self.ops['node_cel_eval'] = self.build_specific_eval_graph_model()
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    def _create_placeholders(self):
            self.placeholders['events_nodes_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event%i_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['events_nodes_type_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event%i_type_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['events_nodes_history_ph'] = [tf.placeholder(tf.int32, shape=[None, self._max_his_num, self._type_num],
                                                        name='event%i_histrory_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['events_partition_idx_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event%i_partition_idx_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['nodes_embedding_ph'] = tf.placeholder(tf.float64,
                                                    shape=[self._num_node, self._h_dim], name='node_embedding_ph')

            self.placeholders['nodes_cellstates_ph'] = tf.placeholder(tf.float64,
                                                    shape=[self._num_node, self._h_dim], name='node_cellstates_ph')

            self.placeholders['negevents_nodes_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                    name='event%i_neg%i_ph'%(event, neg))
                                                    for neg in range(self._neg_num)]
                                                    for event in range(self._eventnum_batch)]

            self.placeholders['negevents_nodes_type_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                    name='event%i_neg%i_type_ph' % (event, neg))
                                                    for neg in range(self._neg_num)]
                                                    for event in range(self._eventnum_batch)]

            self.placeholders['negevents_nodes_history_ph'] = [[tf.placeholder(tf.int32,
                                                        shape=[None, self._max_his_num, self._type_num],
                                                        name='event%i_neg%i_histrory_ph'%(event, neg))
                                                        for neg in range(self._neg_num)]
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['is_train'] = tf.placeholder(tf.bool, name='is_train')

            self.placeholders['is_init'] = tf.placeholder(tf.bool, name='is_init')

            self.placeholders['has_neighbor'] = [tf.placeholder(tf.bool, shape=[None],
                                            name='event%i_hasneighbor_ph'%event)
                                            for event in range(self._eventnum_batch)]

            self.placeholders['has_neighbor_neg'] = [[tf.placeholder(tf.bool, shape=[None],
                                            name='event%i_neg%i_hasneighbor_ph'%(event, neg))
                                            for neg in range(self._neg_num)]
                                            for event in range(self._eventnum_batch)]

            # self.placeholders['nodes_last_time_ph'] = tf.placeholder(tf.float64, shape=[self._num_node],
            #                                             name='nodes_last_time_ph')
            #
            # self.placeholders['events_time_ph'] = [tf.placeholder(tf.float64,
            #                                         shape=[1], name='event%i_time_ph'%event)
            #                                         for event in range(self._eventnum_batch)]

###########test placeholder###########################
            self.placeholders['event_nodes_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='event_eval_ph')

            self.placeholders['event_nodes_type_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='event_type_eval_ph')

            self.placeholders['event_nodes_history_eval_ph'] = tf.placeholder(tf.int32,
                                                        shape=[None, self._max_his_num, self._type_num], name='event_history_eval_ph')

            self.placeholders['event_partition_idx_eval_ph'] = tf.placeholder(tf.int32, shape=[None],
                                                            name='event_partition_idx_eval_ph')

            self.placeholders['negevent_nodes_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='negevent_eval_ph')

            self.placeholders['negevent_nodes_type_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='negevent_type_eval_ph')

            self.placeholders['negevent_nodes_history_eval_ph'] = tf.placeholder(tf.int32,
                                                shape=[None, self._max_his_num, self._type_num], name='negevent_history_eval_ph')

            self.placeholders['has_neighbor_eval'] = tf.placeholder(tf.bool, shape=[None], name='hasneighbor_eval_ph')

            self.placeholders['has_neighbor_neg_eval'] = tf.placeholder(tf.bool, shape=[None], name='hasneighbor_neg_eval_ph')

            # self.placeholders['event_time_eval_ph'] = tf.placeholder(tf.float64, shape=[1], name='event_time_eval_ph')
######################################################

    def _create_variables(self):
        cur_seed = random.getrandbits(32)
        self._embedding_init = tf.get_variable('nodes_embedding_init', shape=[1, self._h_dim],
                                       dtype=tf.float64, trainable=True,
                                       # initializer=tf.random_uniform_initializer(-1, 1, seed=cur_seed))
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=cur_seed))

        self._cellstates_init = tf.get_variable('nodes_cellstates_init', shape=[1, self._h_dim],
                                        dtype=tf.float64, trainable=False,
                                        initializer=tf.zeros_initializer())

        # self._node_embedding_init = tf.get_variable('nodes_embedding_init', shape=[self._num_node, self._h_dim],
        #                                        dtype=tf.float64, trainable=True,
        #                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        # # initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        #
        # self._node_cellstates_init = tf.get_variable('nodes_cellstates_init', shape=[self._num_node, self._h_dim],
        #                                         dtype=tf.float64, trainable=False,
        #                                         initializer=tf.zeros_initializer())

        # self._embedding_st = tf.get_variable('nodes_embedding_st', shape=[self._num_node, self._h_dim],
        #                             dtype=tf.float64, trainable=True,
        #                             initializer=tf.random_uniform_initializer(-1, 1, seed=cur_seed))
        #                             # initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            self.activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            self.activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        if self._aggregator_type == 'mean':
            aggregator = MeanAggregator
        elif self._aggregator_type == 'attention':
            aggregator = AttentionAggregator2
        else:
            raise Exception('Unknown aggregator: ', self._aggregator_type)
        self.weights['aggregator'] = aggregator(self._h_dim, keep=self._keep if self._istraining else 1.)
        self.weights['type_weights_scalar'] = tf.Variable(tf.ones([self._type_num], dtype=tf.float64), trainable=True)
        self.weights['rnn_cells'] = {}
        if self.params['use_different_cell']:
            for node_type in range(self._type_num):
                cell = tf.nn.rnn_cell.LSTMCell(self._h_dim, activation=self.activation_fun)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self._keep if self._istraining else 1.)
                self.weights['rnn_cells'][node_type] = cell
        else:
            cell = tf.nn.rnn_cell.LSTMCell(self._h_dim, activation=self.activation_fun)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self._keep if self._istraining else 1.)
            for node_type in range(self._type_num):
                self.weights['rnn_cells'][node_type] = cell

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._neg_num = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']
        self._num_node = sum(self._num_node_type)
        self._sub_events_train = self.train_data.get_subevents()
        self._sub_events_valid = self.valid_data.get_subevents()
        self._sub_events_test = self.test_data.get_subevents()
        self._max_his_num = self.params['max_his_num']
        self._keep = self.params['keep_prob']
        self._istraining = self.params['is_training']
        self._aggregator_type = self.params['aggregator_type'].lower()

        self._create_placeholders()
        self._create_variables()


    def build_specific_graph_model(self):
        def init_embedding():
            self._node_embedding_init = tf.tile(self._embedding_init, [self._num_node, 1])
            return self._node_embedding_init
        def init_cellstates():
            self._node_cellstates_init = tf.tile(self._cellstates_init, [self._num_node, 1])
            return self._node_cellstates_init
        def assign_embedding():
            return self.placeholders['nodes_embedding_ph']
        def assign_cellstates():
            return self.placeholders['nodes_cellstates_ph']

        node_vec = [None for _ in range(self._eventnum_batch + 1)]
        node_vec[0] = tf.cond(self.placeholders['is_init'], init_embedding, assign_embedding)
        node_cel = [None for _ in range(self._eventnum_batch + 1)]
        node_cel[0] = tf.cond(self.placeholders['is_init'], init_cellstates, assign_cellstates)
        event_pred_list = []
        send_states_list = []
        neg_event_pred_list = []
        neg_send_states_list = []
        # self.concat_layer = Dense(2 * self._h_dim, self._h_dim, act=lambda x: x, keep=self._keep if self._istraining else 1.)
        self.triangularize_layer = Triangularize()
        self.pred_layer = tf.layers.Dense(units=1)
        self.attention_concat_layer = tf.layers.Dense(units=self._h_dim, use_bias=True)
        # self.attention_gate_layer = tf.layers.Dense(units=self._h_dim, activation=tf.nn.sigmoid, use_bias=True)
        for event_id in range(self._eventnum_batch):
            neg_event_states_stacked = [None for _ in range(self._neg_num)]
            neg_send_states = [None for _ in range(self._neg_num)]
            node_vec_part = tf.dynamic_partition(node_vec[event_id],
                                                 self.placeholders['events_partition_idx_ph'][event_id],
                                                 self._type_num + 1)
            node_cel_part = tf.dynamic_partition(node_cel[event_id],
                                                 self.placeholders['events_partition_idx_ph'][event_id],
                                                 self._type_num + 1)
            dy_states = tf.nn.embedding_lookup(node_vec[event_id], self.placeholders['events_nodes_ph'][event_id])
            # st_states = tf.nn.embedding_lookup(self._embedding_st, self.placeholders['events_nodes_ph'][event_id])
            # concat_states = self.concat_layer(tf.concat([st_states, dy_states], 1))
            his_events_states = tf.nn.embedding_lookup(node_vec[event_id], self.placeholders['events_nodes_history_ph'][event_id])
            his_events_states = tf.einsum('nmth,t->nmh', his_events_states, self.weights['type_weights_scalar'])
            his_states = self.weights['aggregator']((dy_states, his_events_states))
            # gate_weight = self.attention_gate_layer(tf.concat([dy_states, his_states], 1))
            # hisconcat_states = tf.add(tf.multiply(gate_weight, dy_states), tf.multiply(1-gate_weight, his_states))
            hisconcat_states = self.attention_concat_layer(tf.concat([dy_states, his_states], 1))
            hisconcat_states = tf.where(self.placeholders['has_neighbor'][event_id], hisconcat_states, dy_states)
            # hisconcat_states = dy_states
            event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['events_nodes_type_ph'][event_id])
            event_states = hisconcat_states*tf.expand_dims(event_weights, 1)
            send_states = tf.reduce_sum(event_states, axis=0)
            for neg in range(self._neg_num):
                neg_dy_states = tf.nn.embedding_lookup(node_vec[event_id], self.placeholders['negevents_nodes_ph'][event_id][neg])
                # neg_st_states = tf.nn.embedding_lookup(self._embedding_st, self.placeholders['negevents_nodes_ph'][event_id][neg])
                # neg_concat_states = self.concat_layer(tf.concat([neg_st_states, neg_dy_states], 1))
                neg_his_events_states = tf.nn.embedding_lookup(node_vec[event_id], self.placeholders['negevents_nodes_history_ph'][event_id][neg])
                neg_his_events_states = tf.einsum('nmth,t->nmh', neg_his_events_states, self.weights['type_weights_scalar'])
                neg_his_states = self.weights['aggregator']((neg_dy_states, neg_his_events_states))
                # neg_gate_weight = self.attention_gate_layer(tf.concat([neg_dy_states, neg_his_states], 1))
                # neg_hisconcat_states = tf.add(tf.multiply(neg_gate_weight, neg_dy_states), tf.multiply(1-neg_gate_weight, neg_his_states))
                neg_hisconcat_states = self.attention_concat_layer(tf.concat([neg_dy_states, neg_his_states], 1))
                neg_hisconcat_states = tf.where(self.placeholders['has_neighbor_neg'][event_id][neg], neg_hisconcat_states, neg_dy_states)
                # neg_hisconcat_states = neg_dy_states
                neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['negevents_nodes_type_ph'][event_id][neg])
                neg_event_states = neg_hisconcat_states*tf.expand_dims(neg_event_weights, 1)
                neg_send_states[neg] = tf.reduce_sum(neg_event_states, axis=0)
                neg_event_states_stacked[neg] = neg_event_states
            send_states_list.append(send_states)
            neg_send_states_list.append(tf.stack(neg_send_states))
            new_target_embedding = [node_vec_part[0]]
            new_target_celstates = [node_cel_part[0]]
            send_states = tf.expand_dims(send_states, 0)
            for node_type in range(self._type_num):
                target_embedding = node_vec_part[node_type+1]
                send_states_pertype = tf.tile(send_states, [tf.shape(target_embedding)[0], 1])
                target_celstates = node_cel_part[node_type+1]
                _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=send_states_pertype,
                                                                    state=(target_celstates, target_embedding))
                new_target_celstates.append(new_target_states_ch[0])
                new_target_embedding.append(new_target_states_ch[1])
            condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                    self.placeholders['events_partition_idx_ph'][event_id], self._type_num+1)
            node_cel[event_id+1] = tf.dynamic_stitch(condition_indices, new_target_celstates)
            node_vec[event_id+1] = tf.dynamic_stitch(condition_indices, new_target_embedding)

        ###dense layer to predict
        # event_scores = tf.stack(send_states_list)
        # neg_event_scores = tf.stack(neg_send_states_list)
        # event_scores = self.pred_layer(event_scores)
        # neg_event_scores = tf.reshape(neg_event_scores, [-1, self._h_dim])
        # neg_event_scores = self.pred_layer(neg_event_scores)
        # event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(event_scores), logits=event_scores)
        # neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_event_scores),
        #                                                            logits=neg_event_scores)
        # loss_mean = (tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses)) / self._eventnum_batch
        ################
        ###learnsuc based to predict
        # event_scores = tf.stack(send_states_list)
        # neg_event_scores = tf.stack(neg_send_states_list)
        # event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores), axis=1))
        # neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores), axis=2))
        # pred = tf.tanh(event_scores_norms / 2)
        # neg_pred = tf.tanh(neg_event_scores_norms / 2)
        # event_losses = tf.log(tf.tanh(event_scores_norms/2))
        # # neg_event_losses = tf.log(1.1-tf.tanh(neg_event_scores_norms/2))
        # neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms)/2))
        # neg_event_losses = tf.reduce_sum(neg_event_losses, axis=1)
        # losses = event_losses + neg_event_losses
        # loss_mean = -tf.reduce_mean(losses)
        #################
        ###pairwise layer to predict
            event_scores = tf.expand_dims(event_states, 0)
            neg_event_scores = tf.stack(neg_event_states_stacked)
            event_scores_h = tf.matmul(event_scores, event_scores, transpose_b=True)
            event_scores_h = self.triangularize_layer(event_scores_h)
            event_scores_h = tf.layers.flatten(event_scores_h)
            y_pred = tf.reduce_sum(event_scores_h, 1)
            neg_event_scores_h = tf.matmul(neg_event_scores, neg_event_scores, transpose_b=True)
            neg_event_scores_h = self.triangularize_layer(neg_event_scores_h)
            neg_event_scores_h = tf.layers.flatten(neg_event_scores_h)
            neg_y_pred = tf.reduce_sum(neg_event_scores_h, 1)
            event_pred_list.append(y_pred)
            neg_event_pred_list.append(neg_y_pred)
        # pred = tf.sigmoid(tf.squeeze(tf.stack(event_pred_list)))
        # neg_pred = tf.sigmoid(tf.squeeze(tf.stack(neg_event_pred_list)))
        pred = event_pred_list
        neg_pred = neg_event_pred_list
        event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pred), logits=pred)
        neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_pred),
                                                                   logits=neg_pred)
        loss_mean = (tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses)) / self._eventnum_batch
        return loss_mean, node_vec[event_id+1], node_cel[event_id+1], tf.sigmoid(tf.squeeze(pred)), tf.sigmoid(tf.squeeze(neg_pred))

    def build_specific_eval_graph_model(self):
        node_vec_part = tf.dynamic_partition(self.placeholders['nodes_embedding_ph'],
                                             self.placeholders['event_partition_idx_eval_ph'], self._type_num + 1)
        node_cel_part = tf.dynamic_partition(self.placeholders['nodes_cellstates_ph'],
                                             self.placeholders['event_partition_idx_eval_ph'], self._type_num + 1)
        dy_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'], self.placeholders['event_nodes_eval_ph'])
        # st_states = tf.nn.embedding_lookup(self._embedding_st, self.placeholders['event_nodes_eval_ph'])
        # concat_states = self.concat_layer(tf.concat([st_states, dy_states], 1))
        his_events_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'],
                                                   self.placeholders['event_nodes_history_eval_ph'])
        his_events_states = tf.einsum('nmth,t->nmh', his_events_states, self.weights['type_weights_scalar'])
        his_states = self.weights['aggregator']((dy_states, his_events_states))
        # gate_weight = self.attention_gate_layer(tf.concat([dy_states, his_states], 1))
        # hisconcat_states = tf.add(tf.multiply(gate_weight, dy_states), tf.multiply(1 - gate_weight, his_states))
        hisconcat_states = self.attention_concat_layer(tf.concat([dy_states, his_states], 1))
        hisconcat_states = tf.where(self.placeholders['has_neighbor_eval'], hisconcat_states, dy_states)
        # hisconcat_states = dy_states
        event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['event_nodes_type_eval_ph'])
        event_states = hisconcat_states * tf.expand_dims(event_weights, 1)
        send_states = tf.reduce_sum(event_states, axis=0)
        ##########neg######
        neg_dy_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'], self.placeholders['negevent_nodes_eval_ph'])
        # neg_st_states = tf.nn.embedding_lookup(self._embedding_st, self.placeholders['negevent_nodes_eval_ph'])
        # neg_concat_states = self.concat_layer(tf.concat([neg_st_states, neg_dy_states], 1))
        neg_his_events_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'],
                                                       self.placeholders['negevent_nodes_history_eval_ph'])
        neg_his_events_states = tf.einsum('nmth,t->nmh', neg_his_events_states, self.weights['type_weights_scalar'])
        neg_his_states = self.weights['aggregator']((neg_dy_states, neg_his_events_states))
        # neg_gate_weight = self.attention_gate_layer(tf.concat([neg_dy_states, neg_his_states], 1))
        # neg_hisconcat_states = tf.add(tf.multiply(neg_gate_weight, neg_dy_states), tf.multiply(1 - neg_gate_weight, neg_his_states))
        neg_hisconcat_states = self.attention_concat_layer(tf.concat([neg_dy_states, neg_his_states], 1))
        neg_hisconcat_states = tf.where(self.placeholders['has_neighbor_neg_eval'], neg_hisconcat_states, neg_dy_states)
        # neg_hisconcat_states = neg_dy_states
        neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['negevent_nodes_type_eval_ph'])
        neg_event_states = neg_hisconcat_states * tf.expand_dims(neg_event_weights, 1)
        neg_send_states = tf.reduce_sum(neg_event_states, axis=0)

        new_target_embedding = [node_vec_part[0]]
        new_target_celstates = [node_cel_part[0]]
        pre_send_states = send_states
        pre_neg_send_states = neg_send_states
        send_states = tf.expand_dims(send_states, 0)
        for node_type in range(self._type_num):
            target_embedding = node_vec_part[node_type+1]
            send_states_pertype = tf.tile(send_states, [tf.shape(target_embedding)[0], 1])
            target_celstates = node_cel_part[node_type+1]
            _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=send_states_pertype,
                                                            state=(target_celstates, target_embedding))
            new_target_celstates.append(new_target_states_ch[0])
            new_target_embedding.append(new_target_states_ch[1])

        condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                                 self.placeholders['event_partition_idx_eval_ph'],
                                                 self._type_num + 1)

        node_cel = tf.dynamic_stitch(condition_indices, new_target_celstates)
        node_vec = tf.dynamic_stitch(condition_indices, new_target_embedding)
        ###dense layer to predict
        # event_scores = tf.expand_dims(pre_send_states,0)
        # neg_event_scores = tf.expand_dims(pre_neg_send_states,0)
        # event_scores = self.pred_layer(event_scores)
        # neg_event_scores = self.pred_layer(neg_event_scores)
        # event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(event_scores), logits=event_scores)
        # neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_event_scores),
        #                                                            logits=neg_event_scores)
        # loss_mean = (tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses))
        # predict = tf.sigmoid(tf.squeeze(event_scores))
        # neg_predict = tf.sigmoid(tf.squeeze(neg_event_scores))
        #######################################
        ###learnsuc based to predict
        # event_scores = send_states
        # neg_event_scores = neg_send_states
        # event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores)))
        # neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores)))
        # predict = tf.tanh(event_scores_norms / 2)
        # neg_predict = tf.tanh(neg_event_scores_norms / 2)
        # event_losses = tf.log(tf.tanh(event_scores_norms / 2))
        # neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms) / 2))
        # # neg_event_losses = tf.log(1-tf.tanh(neg_event_scores_norms / 2))
        # loss_mean = -(event_losses + neg_event_losses)
        #######################################
        ###pairwise layer to predict
        event_scores = event_states
        neg_event_scores = neg_event_states
        event_scores = tf.expand_dims(event_scores, 0)
        neg_event_scores = tf.expand_dims(neg_event_scores, 0)
        event_scores_h = tf.matmul(event_scores, event_scores, transpose_b=True)
        event_scores_h = self.triangularize_layer(event_scores_h)
        event_scores_h = tf.layers.flatten(event_scores_h)
        y_pred = tf.reduce_sum(event_scores_h, 1, keepdims=True)
        neg_event_scores_h = tf.matmul(neg_event_scores, neg_event_scores, transpose_b=True)
        neg_event_scores_h = self.triangularize_layer(neg_event_scores_h)
        neg_event_scores_h = tf.layers.flatten(neg_event_scores_h)
        neg_y_pred = tf.reduce_sum(neg_event_scores_h, 1, keepdims=True)
        event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)
        neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_y_pred),
                                                                   logits=neg_y_pred)
        loss_mean = tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses)
        predict = tf.sigmoid(tf.squeeze(y_pred))
        neg_predict = tf.sigmoid(tf.squeeze(neg_y_pred))
        return loss_mean, predict, neg_predict, node_vec, node_cel


    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        self.node_embedding_cur = np.zeros([self._num_node, self._h_dim], dtype=np.float64)
        self.node_cellstates_cur = np.zeros([self._num_node, self._h_dim], dtype=np.float64)
        print('batches'+str(train_batches_num))
        # print(self._embedding_init.eval(session=self.sess))
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []
                self.node_his_event = {node:[] for node in range(self._num_node)}
                is_init = True
                epoch_flag = False
                print('start epoch %i'%(epoch))
                while not epoch_flag:
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train', is_init)
                    fetches = [self.ops['loss'], self.ops['node_vec'], self.ops['node_cel'], \
                               self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                    cost, self.node_embedding_cur, self.node_cellstates_cur, step, lr, _ =\
                        self.sess.run(fetches, feed_dict=batch_feed_dict)
                    epoch_loss.append(cost)
                    is_init = False
                    if np.isnan(cost):
                        log_out.write('Train ' + str(epoch) + ':Nan error!')
                        print('Train ' + str(epoch) + ':Nan error!')
                        return
                    if step == 1 or step % (self.params['decay_step']/10) == 0:
                        avgc = np.mean(epoch_loss)
                        log_out.write('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        print(cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        sys.stdout.flush()
                print('start valid')
                valid_loss = self.validation()
                log_out.write('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                print('start test')
                self.test()
                if valid_loss < best_loss:
                    best_epoch = epoch
                    best_loss = valid_loss
                    ckpt_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
                    self.saver.save(self.sess, ckpt_path, global_step=step)
                    print('model saved to {}'.format(ckpt_path))
                    log_out.write('model saved to {}'.format(ckpt_path))
                    sys.stdout.flush()
                if epoch-best_epoch >= self.params['patience']:
                    log_out.write('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                    print('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                    break
            log_out.write('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            print('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))


    def validation(self):
        ####train step
        # epoch_flag = False
        # is_init = True
        # while not epoch_flag:
        #    fetches = [self.ops['loss'], self.ops['node_vec'], self.ops['node_cel']]
        #    feed_dict_train, epoch_flag = self.get_batch_feed_dict('train', is_init)
        #    cost, self.node_embedding_cur, self.node_cellstates_cur = self.sess.run(fetches, feed_dict=feed_dict_train)
        #    is_init = False

        ####valid step
        epoch_flag = False
        is_init = False
        valid_loss = []
        valid_batches_num = self.valid_data.get_batch_num()
        print('valid nums:' + str(valid_batches_num))
        labels = []
        val_preds = []
        while not epoch_flag:
            fetches = [self.ops['loss'], self.ops['node_vec'], self.ops['node_cel'], self.ops['pred'], self.ops['neg_pred']]
            feed_dict_valid, epoch_flag = self.get_batch_feed_dict('valid', is_init)
            cost, self.node_embedding_cur, self.node_cellstates_cur, pred, neg_pred = self.sess.run(fetches, feed_dict=feed_dict_valid)
            print(cost)
            valid_loss.append(cost)
            labels.extend(list(pred))
            labels.extend(list(neg_pred[:,0]))
            val_preds.extend([1 for _ in range(64)])
            val_preds.extend([0 for _ in range(64)])
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        # print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)
        print('mae:%f, rmse:%f' % (mae, rmse))
        return np.mean(valid_loss)


    def test(self):
        self.test_data.batch_size = 1
        test_batches_num = self.test_data.get_batch_num()
        print('test nums:'+str(test_batches_num))
        epoch_flag = False
        val_preds = []
        labels = []
        test_loss = []
        while not epoch_flag:
            fetches = [self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict'], self.ops['node_vec_eval'], self.ops['node_cel_eval']]
            feed_dict_test, epoch_flag = self.get_feed_dict_eval()
            cost, predict, neg_predict, self.node_embedding_cur, self.node_cellstates_cur = \
                self.sess.run(fetches, feed_dict=feed_dict_test)
            val_preds.append(predict)
            val_preds.append(neg_predict)
            test_loss.append(cost)
            labels.append(1)
            labels.append(0)
        # precision = metrics.precision_score(labels, val_preds, average=None)
        # recall = metrics.recall_score(labels, val_preds, average=None)
        # f1 = metrics.f1_score(labels, val_preds, average=None)
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        # print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)
        print('mae:%f, rmse:%f' % (mae, rmse))
        print('test cost%f'%(np.mean(test_loss)))

    def sample_negbatch_events(self, batch_data, neg_num):
        batch_data_neg_list = []
        for event in range(len(batch_data)):
            data_neg = [[[] for _ in range(self._type_num+2)] for _ in range(neg_num)]
            for neg in range(neg_num):
                neg_type = random.choice(range(self._type_num))
                for type in range(self._type_num):
                    # if neg_type == type:
                    if True:
                        prenum = 0
                        for pretype in range(type):
                            prenum += self._num_node_type[pretype]
                        while (len(data_neg[neg][type]) < len(batch_data[event][type])):
                            neg_node = random.randint(prenum, prenum+self._num_node_type[type] - 1)
                            if (neg_node in data_neg[neg][type]) or (neg_node in batch_data[event][type]):
                                continue
                            data_neg[neg][type].append(neg_node)
                    else:
                        data_neg[neg][type] = batch_data[event][type]
                data_neg[neg][-1] = batch_data[event][-1]
                data_neg[neg][-2] = batch_data[event][-2]
            batch_data_neg_list.append(data_neg)
        return batch_data_neg_list

    def sample_subevent_fromhis(self, event_data, node_his_event, sub_events_list):
        type_data = []
        for node_type in range(self._type_num):
            type_data.extend(event_data[node_type])
        sampled_his_event = [[] for _ in range(len(type_data))]
        isneighbor = [True for _ in range(len(type_data))]
        for node_index in range(len(type_data)):
            his_len = len(node_his_event[type_data[node_index]])
            if his_len == 0:
                sub_his_event = []
                sub_events = event_data[-2]
                while len(sub_his_event)<self._max_his_num:
                    if len(sub_his_event)+len(sub_events)<self._max_his_num:
                        sub_his_event.extend(sub_events)
                    elif len(sub_his_event)+len(sub_events)==self._max_his_num:
                        sub_his_event.extend(sub_events)
                        break
                    else:
                        extra = len(sub_his_event) + len(sub_events) - self._max_his_num
                        sub_his_event.extend(sub_events[:-extra])
                        break
                sampled_his_event[node_index] = sub_his_event
                isneighbor[node_index] = False
            else:
                sub_his_event = []
                event_his = node_his_event[type_data[node_index]]
                flag = True
                while flag:
                    for his_index in range(len(event_his)):
                        sub_events = sub_events_list[event_his[his_index]]
                        if len(sub_his_event)+len(sub_events)<self._max_his_num:
                            sub_his_event.extend(sub_events)
                        elif len(sub_his_event)+len(sub_events)==self._max_his_num:
                            sub_his_event.extend(sub_events)
                            flag = False
                            break
                        else:
                            extra = len(sub_his_event)+len(sub_events)-self._max_his_num
                            sub_his_event.extend(sub_events[:-extra])
                            flag = False
                            break
                sampled_his_event[node_index] = sub_his_event
                isneighbor[node_index] = True
        return sampled_his_event, isneighbor

    def get_batch_feed_dict(self, state, is_init):
        batch_feed_dict = {}
        batch_feed_dict[self.placeholders['is_init']] = is_init
        if state == 'train':
            batch_data, epoch_flag = self.train_data.next_batch()
            sub_events = self._sub_events_train
        elif state == 'valid':
            batch_data, epoch_flag = self.valid_data.next_batch()
            sub_events_train = self._sub_events_train
            sub_events_valid = self._sub_events_valid
            sub_events = {**sub_events_train, **sub_events_valid}
        else:
            print('state wrong')
        batch_feed_dict[self.placeholders['nodes_embedding_ph']] = self.node_embedding_cur
        batch_feed_dict[self.placeholders['nodes_cellstates_ph']] = self.node_cellstates_cur
        batch_data_neg = self.sample_negbatch_events(batch_data, self._neg_num)
        for event in range(self._eventnum_batch):
            ###############record history event for each node###################
            event_partition = np.zeros(self._num_node, dtype=np.int32)
            event_data = []
            event_data_type = []
            event_data_neg = [[] for _ in range(self._neg_num)]
            event_data_neg_type = [[] for _ in range(self._neg_num)]
            for node_type in range(self._type_num):
                event_data.extend(batch_data[event][node_type])
                event_data_type.extend([node_type]*len(batch_data[event][node_type]))
                event_partition[batch_data[event][node_type]] = node_type + 1
                for node in batch_data[event][node_type]:
                    self.node_his_event[node].append(batch_data[event][-1][0])
                for neg in range(self._neg_num):
                    event_data_neg[neg].extend(batch_data_neg[event][neg][node_type])
                    event_data_neg_type[neg].extend([node_type]*len(batch_data_neg[event][neg][node_type]))
            batch_feed_dict[self.placeholders['events_partition_idx_ph'][event]] = event_partition
            batch_feed_dict[self.placeholders['events_nodes_ph'][event]] = np.asarray(event_data, dtype=np.int32)
            batch_feed_dict[self.placeholders['events_nodes_type_ph'][event]] = np.asarray(event_data_type, dtype=np.int32)
            sampled_his_event, has_neighbor = self.sample_subevent_fromhis(batch_data[event], self.node_his_event, sub_events)
            batch_feed_dict[self.placeholders['events_nodes_history_ph'][event]] = np.asarray(sampled_his_event, dtype=np.int32)
            batch_feed_dict[self.placeholders['has_neighbor'][event]] = has_neighbor
            for neg in range(self._neg_num):
                batch_feed_dict[self.placeholders['negevents_nodes_ph'][event][neg]] = \
                        np.asarray(event_data_neg[neg], dtype=np.int32)
                batch_feed_dict[self.placeholders['negevents_nodes_type_ph'][event][neg]] = \
                        np.asarray(event_data_neg_type[neg], dtype=np.int32)
                sampled_his_event_neg, has_neighbor_neg = self.sample_subevent_fromhis(batch_data_neg[event][neg], self.node_his_event, sub_events)
                batch_feed_dict[self.placeholders['negevents_nodes_history_ph'][event][neg]] = np.asarray(sampled_his_event_neg, dtype=np.int32)
                batch_feed_dict[self.placeholders['has_neighbor_neg'][event][neg]] = has_neighbor_neg
            #############################################
        return batch_feed_dict, epoch_flag

    def get_feed_dict_eval(self):
        feed_dict_eval = {}
        eval_data, epoch_flag = self.test_data.next_batch()
        sub_events_train = self._sub_events_train
        sub_events_valid = self._sub_events_valid
        sub_events_test = self._sub_events_test
        sub_events={**sub_events_train, **sub_events_valid, **sub_events_test}
        feed_dict_eval[self.placeholders['nodes_embedding_ph']] = self.node_embedding_cur
        feed_dict_eval[self.placeholders['nodes_cellstates_ph']] = self.node_cellstates_cur
        eval_data_neg = self.sample_negbatch_events(eval_data, 1)[0][0]
        eval_data = eval_data[0]
        ###################record history event for each node#####################
        event_partition = np.zeros(self._num_node)
        event_data = []
        event_data_type = []
        event_data_neg = []
        event_data_neg_type = []
        for node_type in range(self._type_num):
            event_data.extend(eval_data[node_type])
            event_data_type.extend([node_type]*len(eval_data[node_type]))
            event_partition[eval_data[node_type]] = node_type+1
            event_data_neg.extend(eval_data_neg[node_type])
            event_data_neg_type.extend([node_type]*len(eval_data_neg[node_type]))
            for node in eval_data[node_type]:
                self.node_his_event[node].append(eval_data[-1][0])
        feed_dict_eval[self.placeholders['event_partition_idx_eval_ph']] = event_partition
        feed_dict_eval[self.placeholders['event_nodes_eval_ph']] = np.asarray(event_data, dtype=np.int32)
        feed_dict_eval[self.placeholders['event_nodes_type_eval_ph']] = np.asarray(event_data_type, dtype=np.int32)
        feed_dict_eval[self.placeholders['negevent_nodes_eval_ph']] = np.asarray(event_data_neg, dtype=np.int32)
        feed_dict_eval[self.placeholders['negevent_nodes_type_eval_ph']] = np.asarray(event_data_neg_type, dtype=np.int32)
        sampled_his_event, has_neighbor = self.sample_subevent_fromhis(eval_data, self.node_his_event, sub_events)
        sampled_his_event_neg, has_neighbor_neg = self.sample_subevent_fromhis(eval_data_neg, self.node_his_event, sub_events)
        feed_dict_eval[self.placeholders['event_nodes_history_eval_ph']] = np.asarray(sampled_his_event, dtype=np.int32)
        feed_dict_eval[self.placeholders['has_neighbor_eval']] = has_neighbor
        feed_dict_eval[self.placeholders['negevent_nodes_history_eval_ph']] = np.asarray(sampled_his_event_neg, dtype=np.int32)
        feed_dict_eval[self.placeholders['has_neighbor_neg_eval']] = has_neighbor_neg
        #############################################
        return feed_dict_eval, epoch_flag








