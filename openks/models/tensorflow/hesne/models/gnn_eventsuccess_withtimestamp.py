import sys
import random
import numpy as np
import tensorflow as tf
from utils.inits import glorot_init
import sklearn.metrics as metrics
from models.basic_model import BasicModel
from layers.layers import Dense, EventLayer, TimeDymLayer
from utils.data_manager import *
import scipy.sparse as sp
import time

class GnnEventModel_timestamp(BasicModel):
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
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'changed_eventgate'
        return log_file

    def get_checkpoint_dir(self, params):
        checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'changed_eventgate'
        return checkpoint_dir

    def make_model(self):
        #compute loss
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['loss'], self.ops['node_vec'], self.ops['node_cel'], self.ops['node_last_time'], self.ops['test0'], self.ops['test1'] = self.build_specific_graph_model()
            self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict'], self.ops['node_vec_eval'],\
                self.ops['node_cel_eval'], self.ops['node_last_time_eval'] = self.build_specific_eval_graph_model()
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    def _create_placeholders(self):
            self.placeholders['events_nodes_type_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                    name='node_type%i_event%i_ph' % (node_type, event))
                                                    for node_type in range(self._type_num)]
                                                    for event in range(self._eventnum_batch)]

            self.placeholders['event_partition_idx_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event_partition_idx_event%i_ph' % event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['negevents_nodes_type_ph'] = [[[tf.placeholder(tf.int32, shape=[None],
                                                    name='node_type%i_neg%i_event%i_ph' % (node_type, neg, event))
                                                    for node_type in range(self._type_num)]
                                                    for neg in range(self._neg_num)]
                                                    for event in range(self._eventnum_batch)]

            self.placeholders['node_embedding_ph'] = tf.placeholder(tf.float64,
                                                    shape=[self._num_node, self._h_dim], name='node_embedding_ph')

            self.placeholders['node_cellstates_ph'] = tf.placeholder(tf.float64,
                                                    shape=[self._num_node, self._h_dim], name='node_cellstates_ph')

            self.placeholders['is_train'] = tf.placeholder(tf.bool, name='is_train')

            self.placeholders['is_init'] = tf.placeholder(tf.bool, name='is_init')

            self.placeholders['is_neighbor'] = [tf.placeholder(tf.bool, name='is_neighbor')
                                                    for event in range(self._eventnum_batch)]

            # self.placeholders['neighbor_type'] = [tf.placeholder(tf.float64, shape=[None, self._type_num],
            #                                     name='neighbor_type_event%i' % (event))
            #                                     for event in range(self._eventnum_batch)]

            # self.placeholders['cur_adj_ph'] = [tf.sparse.placeholder(tf.float64, shape=[None, None])
            #                                    for event in range(self._eventnum_batch)]

            # self.placeholders['cur_neighbor_ph'] = [tf.placeholder(tf.int32, shape=[None],
            #                                         name='neighbors_event%i_ph'%(event))
            #                                         for event in range(self._eventnum_batch)]

            self.placeholders['neighbor_partition_idx_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='neighbor_partition_idx_event%i_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['nodes_last_time_ph'] = tf.placeholder(tf.float64, shape=[self._num_node],
                                                        name='nodes_last_time_ph')

            self.placeholders['events_time_ph'] = [tf.placeholder(tf.float64,
                                                    shape=[1], name='event%i_time_ph'%event)
                                                    for event in range(self._eventnum_batch)]

###########test placeholder###########################
            self.placeholders['events_nodes_type_eval_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                            name='node_type%i_eval_ph' % node_type)
                                                            for node_type in range(self._type_num)]

            self.placeholders['event_partition_idx_eval_ph'] = tf.placeholder(tf.int32, shape=[None],
                                                            name='event_partition_idx_eval_ph')

            self.placeholders['negevents_nodes_type_eval_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                            name='neg_node_type%i_eval_ph' % node_type)
                                                            for node_type in range(self._type_num)]

            # self.placeholders['cur_adj_eval_ph'] = tf.sparse.placeholder(tf.float64, shape=[None, None])

            # self.placeholders['cur_neighbor_eval_ph'] = tf.placeholder(tf.int32, shape=[None])

            self.placeholders['is_neighbor_eval'] = tf.placeholder(tf.bool, name='is_neighbor_eval')

            self.placeholders['neighbor_partition_idx_eval_ph'] = tf.placeholder(tf.int32, shape=[None])

            self.placeholders['events_time_eval_ph'] = tf.placeholder(tf.float64, shape=[1], name='event_time_eval_ph')

            # self.placeholders['neighbor_type_eval'] = tf.placeholder(tf.float64, shape=[None, self._type_num],
            #                                                          name='neighbor_type_eval')

######################################################

    def _create_variables(self):
        cur_seed = random.getrandbits(32)

        # self._embedding_init = tf.get_variable('node_embedding_init', shape=[1, self._h_dim],
        #                                dtype=tf.float64, trainable=False,
        #                                 initializer=tf.zeros_initializer())
                                       # initializer=tf.random_uniform_initializer(-1, 1, seed=cur_seed))
                                       # initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

        # self._cellstates_init = tf.get_variable('node_cellstates_init', shape=[1, self._h_dim],
        #                                 dtype=tf.float64, trainable=False,
        #                                 initializer=tf.zeros_initializer())

        self._node_embedding_dy = tf.get_variable('node_embedding_dy', shape=[self._num_node, self._h_dim],
                                            dtype=tf.float64, trainable=True,
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

        self._node_cellstates_dy = tf.get_variable('node_cellstates_dy', shape=[self._num_node, self._h_dim],
                                            dtype=tf.float64, trainable=False,
                                            initializer=tf.zeros_initializer())

        self._node_embedding_st = tf.get_variable('node_embedding_st', shape=[self._num_node, self._h_dim],
                                    dtype=tf.float64, trainable=True,
                                    # initializer=tf.random_uniform_initializer(-1, 1, seed=cur_seed))
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._neg_num = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']
        self._num_node = sum(self._num_node_type)
        # self._max_diffusion_step = self.params['max_diffusion_step']
        # self._eventclass = self.hegraph.get_eventtype_num()
        self._keep = self.params['keep_prob']
        self._istraining = self.params['is_training']

        self._create_placeholders()
        self._create_variables()

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            self.activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
           self.activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        # self.weights['type_weights'] = [tf.Variable(glorot_init([self._h_dim, self._h_dim],
        #                                 name='gnn_type_weights_typ%i' % node_type))
        #                                 for node_type in range(self._type_num)]

        self.weights['type_weights_scalar'] = [tf.Variable(1.0, dtype=tf.float64, trainable=True) for node_type in range(self._type_num)]
        # self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._eventclass]))
        # self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._h_dim]))

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

    def build_specific_graph_model(self):
        def init_embedding():
            return self._node_embedding_dy
        def init_cellstates():
            return self._node_cellstates_dy
        def assign_embedding():
            return self.placeholders['node_embedding_ph']
        def assign_cellstates():
            return self.placeholders['node_cellstates_ph']
        def no_propagation(node_vec, node_cel):
            return node_vec, node_cel
        def event_propagation(event_id, node_vec, node_cel, event_states, new_target_embedding):
            # nonlocal new_target_embedding
            type_ws = [tf.tile([self.weights['type_weights_scalar'][node_type]],
                               [tf.shape(self.placeholders['events_nodes_type_ph'][event_id][node_type])[0]])
                                for node_type in range(self._type_num)]
            type_w = tf.concat(type_ws, axis=0)
            type_wt = tf.diag(type_w)
            send_embedding = tf.concat(new_target_embedding[1:], axis=0)
            neighbor_embedding_part = tf.dynamic_partition(node_vec,
                                                      self.placeholders['neighbor_partition_idx_ph'][event_id], 2)
            neighbor_celstates_part = tf.dynamic_partition(node_cel,
                                                      self.placeholders['neighbor_partition_idx_ph'][event_id], 2)
            target_info = tf.matmul(type_wt, send_embedding)
            target_input = tf.sparse_tensor_dense_matmul(self.placeholders['cur_adj_ph'][event_id], target_info)
            event_states = tf.tile(event_states, [tf.shape(neighbor_embedding_part[1])[0], 1])
            new_neighbor_embedding, new_neighbor_celstates = self.event_layer([event_states, target_input,
                                                                    neighbor_embedding_part[1], neighbor_celstates_part[1]])
            neighbor_condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                                        self.placeholders['neighbor_partition_idx_ph'][event_id], 2)
            new_node_vec = tf.dynamic_stitch(neighbor_condition_indices,
                                                       [neighbor_embedding_part[0], new_neighbor_embedding])
            new_node_cel = tf.dynamic_stitch(neighbor_condition_indices,
                                                       [neighbor_celstates_part[0], new_neighbor_celstates])
            return new_node_vec, new_node_cel

        node_vec = [None for event in range(self._eventnum_batch + 1)]
        node_vec[0] = tf.cond(self.placeholders['is_init'], init_embedding, assign_embedding)
        node_cel = [None for event in range(self._eventnum_batch + 1)]
        node_cel[0] = tf.cond(self.placeholders['is_init'], init_cellstates, assign_cellstates)
        node_last_time = [None for event in range(self._eventnum_batch + 1)]
        node_last_time[0] = self.placeholders['nodes_last_time_ph']
        event_states_list = []
        neg_event_states_list = []
        self.dense_layer = Dense(2 * self._h_dim, self._h_dim, act=lambda x: x, keep=self._keep if self._istraining else 1.)
        #######add pred_layer
        self.pred_layer = Dense(self._h_dim, 1, act=lambda x: x, keep=self._keep if self._istraining else 1.)
        self.event_layer = EventLayer(self._h_dim, keep=self._keep if self._istraining else 1.)
        self.timedyn_layer = TimeDymLayer(self._h_dim, keep=self._keep if self._istraining else 1.)
        for event_id in range(self._eventnum_batch):
            event_states = tf.zeros([1, self._h_dim], dtype=tf.float64)
            neg_event_states = [tf.zeros([1, self._h_dim], dtype=tf.float64) for _ in range(self._neg_num)]
            node_vec_part = tf.dynamic_partition(node_vec[event_id],
                                                 self.placeholders['event_partition_idx_ph'][event_id],
                                                 self._type_num + 1)
            node_cel_part = tf.dynamic_partition(node_cel[event_id],
                                                 self.placeholders['event_partition_idx_ph'][event_id],
                                                 self._type_num + 1)
            node_last_time_part = tf.dynamic_partition(node_last_time[event_id],
                                                       self.placeholders['event_partition_idx_ph'][event_id],
                                                       self._type_num + 1)
            for node_type in range(self._type_num):
                states_st = tf.nn.embedding_lookup(self._node_embedding_st,
                                    self.placeholders['events_nodes_type_ph'][event_id][node_type])
                states_dy = node_vec_part[node_type+1]
                ####################concat embedding################
                concate_states = tf.concat([states_st, states_dy], 1)
                states = self.dense_layer(concate_states)
                ## states = states_dy_pertype
                ####################embedding with gate#############
                # last_time = tf.expand_dims(node_last_time_part[node_type+1], 1)
                # event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_ph'][event_id], 1),
                #                      [tf.shape(last_time)[0], 1])
                # delta_time = event_time - last_time
                # states = self.timedyn_layer([delta_time, states_st, states_dy])
                #####################################################
                states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
                # event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])
                event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], states_aggregated)
            # event_states = self.activation_fun(tf.matmul(event_states, self.weights['event_weights'])) + event_states
            for neg in range(self._neg_num):
                for node_type in range(self._type_num):
                    neg_states_dy = tf.nn.embedding_lookup(node_vec[event_id],
                                        self.placeholders['negevents_nodes_type_ph'][event_id][neg][node_type])
                    neg_states_st = tf.nn.embedding_lookup(self._node_embedding_st,
                                        self.placeholders['negevents_nodes_type_ph'][event_id][neg][node_type])
                    #################concat embedding##################
                    neg_concate_states = tf.concat([neg_states_st, neg_states_dy], 1)
                    neg_states = self.dense_layer(neg_concate_states)
                    ## neg_states = neg_states_dy
                    #################embedding with gate###############
                    # neg_last_time = tf.expand_dims(tf.nn.embedding_lookup(node_last_time[event_id],
                    #                     self.placeholders['negevents_nodes_type_ph'][event_id][neg][node_type]), 1)
                    # neg_event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_ph'][event_id], 1),
                    #                  [tf.shape(neg_last_time)[0], 1])
                    # neg_delta_time = neg_event_time - neg_last_time
                    # neg_states = self.timedyn_layer([neg_delta_time, neg_states_st, neg_states_dy])
                    ###################################################
                    neg_states_aggregated = tf.reduce_mean(neg_states, axis=0, keepdims=True)
                    # neg_event_states[neg] += tf.matmul(neg_states_aggregated, self.weights['type_weights'][node_type])
                    neg_event_states[neg] += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], neg_states_aggregated)
                # neg_event_states[neg] = self.activation_fun(tf.matmul(neg_event_states[neg], self.weights['event_weights'])) + neg_event_states[neg]
            neg_event_states = tf.stack(neg_event_states)
            event_states_list.append(tf.squeeze(event_states))
            neg_event_states_list.append(tf.squeeze(neg_event_states))
            # neg_event_states_list.append([tf.squeeze(neg_event_states[neg]) for neg in range(self._neg_num)])

            new_target_embedding = [node_vec_part[0]]
            new_target_celstates = [node_cel_part[0]]
            new_node_last_time = [node_last_time_part[0]]

            for node_type in range(self._type_num):
                target_embedding = node_vec_part[node_type+1]
                sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
                target_celstates = node_cel_part[node_type+1]
                _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=sent_states,
                                                                    state=(target_celstates, target_embedding))
                new_target_celstates.append(new_target_states_ch[0])
                new_target_embedding.append(new_target_states_ch[1])
                new_last_time = tf.tile(self.placeholders['events_time_ph'][event_id],[tf.shape(target_embedding)[0]])
                new_node_last_time.append(new_last_time)
            condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                    self.placeholders['event_partition_idx_ph'][event_id], self._type_num+1)

            node_cel[event_id+1] = tf.dynamic_stitch(condition_indices, new_target_celstates)
            node_vec[event_id+1] = tf.dynamic_stitch(condition_indices, new_target_embedding)
            node_last_time[event_id+1] = tf.dynamic_stitch(condition_indices, new_node_last_time)

            #####states propagation#############
            # node_vec[event_id+1], node_cel[event_id+1] = tf.cond(self.placeholders['is_neighbor'][event_id],
            #                     lambda: event_propagation(event_id, node_vec[event_id+1], node_cel[event_id+1], event_states, new_target_embedding),
            #                     lambda: no_propagation(node_vec[event_id+1], node_cel[event_id+1]))
            #####################################

        event_scores = tf.stack(event_states_list)
        neg_event_scores = tf.stack(neg_event_states_list)

        #############
        event_scores = self.pred_layer(event_scores)
        neg_event_scores = tf.reshape(neg_event_scores, [-1, self._h_dim])
        neg_event_scores = self.pred_layer(neg_event_scores)
        event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(event_scores), logits=event_scores)
        neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_event_scores), logits=neg_event_scores)
        loss_mean = (tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses)) / self._eventnum_batch
        # event_losses = tf.log(event_scores)
        # neg_event_losses = (1-neg_event_scores)



        # event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores), axis=1))
        # neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores), axis=2))
        #
        # # predict = tf.tanh(event_scores_norms/2)
        # # neg_predict = tf.tanh(neg_event_scores_norms/2)
        #
        # event_losses = tf.log(tf.tanh(event_scores_norms/2))
        # neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms)/2))
        # neg_event_losses = tf.reduce_sum(neg_event_losses, axis=1)
        # losses = event_losses + neg_event_losses
        # loss_mean = -tf.reduce_mean(losses)
        return loss_mean, node_vec[event_id+1], node_cel[event_id+1], node_last_time[event_id+1], tf.reduce_mean(event_losses), tf.reduce_mean(neg_event_losses)

    def build_specific_eval_graph_model(self):
        def no_propagation(node_vec, node_cel):
            return node_vec, node_cel
        def event_propagation(node_vec, node_cel, event_states, new_target_embedding):
            # nonlocal new_target_embedding
            type_ws = [tf.tile([self.weights['type_weights_scalar'][node_type]],
                               [tf.shape(self.placeholders['events_nodes_type_eval_ph'][node_type])[0]])
                                for node_type in range(self._type_num)]
            type_w = tf.concat(type_ws, axis=0)
            type_wt = tf.diag(type_w)
            send_embedding = tf.concat(new_target_embedding[1:], axis=0)
            neighbor_embedding_part = tf.dynamic_partition(node_vec,
                                                      self.placeholders['neighbor_partition_idx_eval_ph'], 2)
            neighbor_celstates_part = tf.dynamic_partition(node_cel,
                                                      self.placeholders['neighbor_partition_idx_eval_ph'], 2)
            target_info = tf.matmul(type_wt, send_embedding)
            target_input = tf.sparse_tensor_dense_matmul(self.placeholders['cur_adj_eval_ph'], target_info)
            event_states = tf.tile(event_states, [tf.shape(neighbor_embedding_part[1])[0], 1])
            new_neighbor_embedding, new_neighbor_celstates = self.event_layer([event_states, target_input,
                                                                neighbor_embedding_part[1], neighbor_celstates_part[1]])
            neighbor_condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                                        self.placeholders['neighbor_partition_idx_eval_ph'], 2)
            new_node_vec = tf.dynamic_stitch(neighbor_condition_indices,
                                                       [neighbor_embedding_part[0], new_neighbor_embedding])
            new_node_cel = tf.dynamic_stitch(neighbor_condition_indices,
                                                       [neighbor_celstates_part[0], new_neighbor_celstates])
            return new_node_vec, new_node_cel

        event_states = tf.zeros([1, self._h_dim], dtype=tf.float64)
        neg_event_states = tf.zeros([1, self._h_dim], dtype=tf.float64)
        node_vec_part = tf.dynamic_partition(self.placeholders['node_embedding_ph'],
                                             self.placeholders['event_partition_idx_eval_ph'],
                                             self._type_num + 1)
        node_cel_part = tf.dynamic_partition(self.placeholders['node_cellstates_ph'],
                                             self.placeholders['event_partition_idx_eval_ph'],
                                             self._type_num + 1)
        node_last_time_part = tf.dynamic_partition(self.placeholders['nodes_last_time_ph'],
                                            self.placeholders['event_partition_idx_eval_ph'],
                                            self._type_num + 1)
        for node_type in range(self._type_num):
            states_dy = node_vec_part[node_type+1]
            states_st = tf.nn.embedding_lookup(self._node_embedding_st,
                                               self.placeholders['events_nodes_type_eval_ph'][node_type])
            ##########concat embedding##########
            concate_states = tf.concat([states_st, states_dy], 1)
            states = self.dense_layer(concate_states)
            ##########embedding with gate#######
            # last_time = tf.expand_dims(tf.nn.embedding_lookup(self.placeholders['nodes_last_time_ph'],
            #                             self.placeholders['events_nodes_type_eval_ph'][node_type]), 1)
            # event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_eval_ph'], 1),
            #                      [tf.shape(last_time)[0], 1])
            # delta_time = event_time - last_time
            # states = self.timedyn_layer([delta_time, states_st, states_dy])
            ## event_types = self.timedyn_layer([delta_time, states_st, states_dy])
            ####################################
            states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
            # event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])
            event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], states_aggregated)
            # event_states = self.activation_fun(tf.matmul(event_states, self.weights['event_weights'])) + event_states
            neg_states_dy = tf.nn.embedding_lookup(self.placeholders['node_embedding_ph'],
                                                   self.placeholders['negevents_nodes_type_eval_ph'][node_type])
            neg_states_st = tf.nn.embedding_lookup(self._node_embedding_st,
                                                   self.placeholders['negevents_nodes_type_eval_ph'][node_type])
            #############concat negembedding##############
            neg_concate_states = tf.concat([neg_states_st, neg_states_dy], 1)
            neg_states = self.dense_layer(neg_concate_states)
            #############embedding with gate#############
            # neg_last_time = tf.expand_dims(tf.nn.embedding_lookup(self.placeholders['nodes_last_time_ph'],
            #                                 self.placeholders['negevents_nodes_type_eval_ph'][node_type]), 1)
            # neg_event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_eval_ph'], 1),
            #                      [tf.shape(neg_last_time)[0], 1])
            # neg_delta_time = neg_event_time - neg_last_time
            # neg_states = self.timedyn_layer([neg_delta_time, neg_states_st, neg_states_dy])
            ###############################################
            neg_states_aggregated = tf.reduce_mean(neg_states, axis=0, keepdims=True)
            # neg_event_states[neg] += tf.matmul(neg_states_aggregated, self.weights['type_weights'][node_type])
            neg_event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], neg_states_aggregated)
                # neg_event_states[neg] = self.activation_fun(tf.matmul(neg_event_states[neg], self.weights['event_weights'])) + neg_event_states[neg]

        new_target_embedding = [node_vec_part[0]]
        new_target_celstates = [node_cel_part[0]]
        new_node_last_time = [node_last_time_part[0]]

        for node_type in range(self._type_num):
            target_embedding = node_vec_part[node_type+1]
            sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
            target_celstates = node_cel_part[node_type+1]
            _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=sent_states,
                                                            state=(target_celstates, target_embedding))
            new_target_celstates.append(new_target_states_ch[0])
            new_target_embedding.append(new_target_states_ch[1])
            new_last_time = tf.tile(self.placeholders['events_time_eval_ph'], [tf.shape(target_embedding)[0]])
            new_node_last_time.append(new_last_time)

        condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                                 self.placeholders['event_partition_idx_eval_ph'],
                                                 self._type_num + 1)

        node_cel = tf.dynamic_stitch(condition_indices, new_target_celstates)
        node_vec = tf.dynamic_stitch(condition_indices, new_target_embedding)
        node_last_time = tf.dynamic_stitch(condition_indices, new_node_last_time)

        #####states propagation#############
        # node_vec, node_cel = tf.cond(self.placeholders['is_neighbor_eval'],
        #                              lambda: event_propagation(node_vec, node_cel, event_states, new_target_embedding),
        #                              lambda: no_propagation(node_vec, node_cel))
        ####################################

        # event_scores = tf.squeeze(event_states)
        # neg_event_scores = tf.squeeze(neg_event_states)

        #################
        event_scores = self.pred_layer(event_states)
        neg_event_scores = self.pred_layer(neg_event_states)
        event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(event_scores), logits=event_scores)
        neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_event_scores),
                                                                   logits=neg_event_scores)
        loss_mean = (tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses))
        predict = tf.sigmoid(tf.squeeze(event_scores))
        neg_predict = tf.sigmoid(tf.squeeze(neg_event_scores))



        # event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores)))
        # neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores)))
        #
        # predict = tf.tanh(event_scores_norms / 2)
        # neg_predict = tf.tanh(neg_event_scores_norms / 2)
        #
        # event_losses = tf.log(tf.tanh(event_scores_norms / 2))
        # neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms) / 2))
        # loss_mean = -(event_losses + neg_event_losses)
        return loss_mean, predict, neg_predict, node_vec, node_cel, node_last_time


    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        print('batches'+str(train_batches_num))
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []
                self.node_embedding_cur = np.zeros([self._num_node, self._h_dim], dtype=np.float64)
                self.node_cellstates_cur = np.zeros([self._num_node, self._h_dim], dtype=np.float64)
                self.node_last_time = np.zeros(self._num_node, dtype=np.float64)
                self.cur_adj = np.zeros([self._num_node, self._num_node])
                epoch_flag = False
                print('start epoch %i'%(epoch))
                while not epoch_flag:
                    is_init = True
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train', is_init)
                    fetches = [self.ops['loss'], self.ops['node_vec'], self.ops['node_cel'], self.ops['node_last_time'], self.ops['test0'], self.ops['test1'], \
                               self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                    cost, self.node_embedding_cur, self.node_cellstates_cur, self.node_last_time, test0, test1, step, lr, _ = \
                        self.sess.run(fetches, feed_dict=batch_feed_dict)
                    # print(test0)
                    # print('************')
                    # print(test1)
                    # time.sleep(5)
                    # self.test()
                    epoch_loss.append(cost)
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
                    # if step == 1:
                    #     valid_loss = self.validation()
                    #     print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        # self.test()
                        # self.test()
                        # sys.stdout.flush()
                # if step == 1 or step % self.params['eval_point'] == 0:
                self.test()
                print('start valid')
                valid_loss = self.validation()
                log_out.write('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
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
            print('start test')
            # self.test()


    def validation(self):
        valid_loss = []
        # self.valid_data.batch_size = 1
        valid_batches_num = self.valid_data.get_batch_num()
        print('valid nums:' + str(valid_batches_num))
        # epoch_flag = False
        # for node_type in range(self._type_num):
        # self.node_embedding_evalcur = self._node_embedding_init.eval(session=self.sess)
        # if self.params['graph_rnn_cell'].lower() == 'lstm':
        # self.node_cellstates_evalcur = self._node_cellstates_init.eval(session=self.sess)
        ############train step to get current embedding###########
        is_init = True
        epoch_flag = False
        num = 0
        while not epoch_flag:
            batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train', is_init)
            fetches = [self.ops['node_vec'], self.ops['node_cel'], self.ops['node_last_time'], self.ops['test0'], self.ops['test1']]
            self.node_embedding_cur, self.node_cellstates_cur, self.node_last_time, test0, test1 = \
                self.sess.run(fetches, feed_dict=batch_feed_dict)
            if num%100 == 0:
                print(test0)
                print(test1)
            num+=1
            is_init = False
        ################valid###################
        # is_init = False
        # is_init = True
        epoch_flag = False
        # self.node_embedding_cur = self._node_embedding_dy.eval(session=self.sess)
        # self.node_cellstates_cur = self._node_cellstates_dy.eval(session=self.sess)
        while not epoch_flag:
            fetches = [self.ops['loss'], self.ops['node_vec'], self.ops['node_cel'], self.ops['node_last_time'],self.ops['test0'], self.ops['test1']]
            feed_dict_valid, epoch_flag = self.get_batch_feed_dict('valid', is_init)
            cost, self.node_embedding_cur, self.node_cellstates_cur, self.node_last_time, test0, test1 = \
                                                                self.sess.run(fetches, feed_dict=feed_dict_valid)
            valid_loss.append(cost)
            # print(test0)
            # print('************')
            # print(test1)
        return np.mean(valid_loss)


    def test(self):
        ############train step to get current embedding###########
        # is_init = True
        # while not epoch_flag:
        #     batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train', is_init)
        #     fetches = [self.ops['node_vec'], self.ops['node_cel'], self.ops['node_last_time']]
        #     self.node_embedding_cur, self.node_cellstates_cur, self.node_last_time = \
        #         self.sess.run(fetches, feed_dict=batch_feed_dict)
        #     is_init = False
        # ################valid step to get current embedding###################
        # is_init = False
        # epoch_flag = False
        # while not epoch_flag:
        #     fetches = [self.ops['node_vec'], self.ops['node_cel'], self.ops['node_last_time']]
        #     feed_dict_valid, epoch_flag = self.get_batch_feed_dict('valid', is_init)
        #     self.node_embedding_cur, self.node_cellstates_cur, self.node_last_time = \
        #         self.sess.run(fetches, feed_dict=feed_dict_valid)
        ##################test################################
        self.test_data.batch_size = 1
        test_batches_num = self.test_data.get_batch_num()
        print('test nums:'+str(test_batches_num))
        epoch_flag = False
        val_preds = []
        labels = []
        test_loss = []
        # self.node_embedding_cur = self._node_embedding_dy.eval(session=self.sess)
        # self.node_cellstates_cur = self._node_cellstates_dy.eval(session=self.sess)
        while not epoch_flag:
            # test_data_neg = test_data_neg_list[num]
            fetches = [self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict'], self.ops['node_vec_eval'], self.ops['node_cel_eval'], \
                       self.ops['node_last_time_eval']]
            feed_dict_test, epoch_flag = self.get_feed_dict_eval()
            cost, predict, neg_predict, self.node_embedding_cur, self.node_cellstates_cur, self.node_last_time = \
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


    def sample_neg_events(self, batch_data, neg_num):
        batch_data_neg_list = []
        for event in range(len(batch_data)):
            data_neg = [[[] for type in range(self._type_num)] for neg in range(neg_num)]
            for neg in range(neg_num):
                for type in range(self._type_num):
                    while (len(data_neg[neg][type]) < len(batch_data[event][type])):
                        neg_node = random.randint(0, self._num_node_type[type]-1)
                        if neg_node in data_neg[neg][type]:
                            continue
                        data_neg[neg][type].append(neg_node)
            batch_data_neg_list.append(data_neg)
        return batch_data_neg_list


    def sparse_to_tuple(self, matrix):
        matrix = sp.coo_matrix(matrix)
        if sp.isspmatrix_coo(matrix):
            # matrix = matrix.tocoo()
            coords = np.vstack((matrix.row, matrix.col)).transpose()
            values = matrix.data
            shape = matrix.shape
            return coords, values, shape

    def get_batch_feed_dict(self, state, is_init):
        batch_feed_dict = {}
        batch_feed_dict[self.placeholders['is_init']] = is_init
        if state == 'train':
            batch_data, epoch_flag = self.train_data.next_batch()
        elif state == 'valid':
            batch_data, epoch_flag = self.valid_data.next_batch()
        else:
            print('state wrong')
        batch_feed_dict[self.placeholders['node_embedding_ph']] = self.node_embedding_cur
        batch_feed_dict[self.placeholders['node_cellstates_ph']] = self.node_cellstates_cur
        batch_data_neg_list = self.sample_neg_events(batch_data, self._neg_num)
        for event in range(self._eventnum_batch):
            ###############feed neighbor###################
            node_list = []
            for node_type in range(self._type_num):
                previous_num = 0
                for type in range(node_type):
                    previous_num += self._num_node_type[type]
                batch_data[event][node_type] = [node+previous_num for node in batch_data[event][node_type]]
                node_list += batch_data[event][node_type]
            for nodei in node_list:
                for nodej in node_list:
                    if nodei != nodej:
                        self.cur_adj[nodei][nodej] = 0
            neighbor_nodes = []
            for node in node_list:
                neighbor_nodes.extend(list(np.where(self.cur_adj[node]>0)[0]))
            if len(neighbor_nodes) == 0:
                batch_feed_dict[self.placeholders['is_neighbor'][event]] = False
                neighbor_nodes = node_list
            else:
                batch_feed_dict[self.placeholders['is_neighbor'][event]] = True
                neighbor_nodes = sorted(np.unique(neighbor_nodes))
            ############neighbor link type########
            # neighbor_type = [[0]*self._type_num]*len(neighbor_nodes)
            # for index in range(len(neighbor_nodes)):
            #     for node_type in range(self._type_num):
            #         neighbor_type[index][node_type] = sum(self.cur_adj[neighbor_nodes[index]][batch_data[event][node_type]])
            # batch_feed_dict[self.placeholders['neighbor_type'][event]] = np.asarray(neighbor_type, dtype=np.int32)
            ###############feed adj#######################
            # neighbors = self.cur_adj[np.ix_(neighbor_nodes, node_list)]
            # neighbor_indicese, neighbor_values, neighbor_shape = self.sparse_to_tuple(neighbors)
            # batch_feed_dict[self.placeholders['cur_adj_ph'][event]] = (neighbor_indicese, neighbor_values, neighbor_shape)
            for nodei in node_list:
                for nodej in node_list:
                    if nodei != nodej:
                        self.cur_adj[nodei][nodej] = 1
            neighbor_partition = np.zeros(self._num_node, dtype=np.int32)
            neighbor_partition[neighbor_nodes] = 1
            batch_feed_dict[self.placeholders['neighbor_partition_idx_ph'][event]] = neighbor_partition
            ##################feed event time############
            batch_feed_dict[self.placeholders['nodes_last_time_ph']] = self.node_last_time
            batch_feed_dict[self.placeholders['events_time_ph'][event]] = batch_data[event][-1]
            #############################################
            event_partition = np.zeros(self._num_node, dtype=np.int32)
            # neg_event_partition = [np.zeros(self._num_node, dtype=np.int32) for neg in range(self._neg_num)]
            for node_type in range(self._type_num):
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] =\
                    np.asarray(batch_data[event][node_type], dtype=np.int32)
                event_partition[batch_data[event][node_type]] = node_type+1
            batch_feed_dict[self.placeholders['event_partition_idx_ph'][event]] = event_partition
            for neg in range(self._neg_num):
                for node_type in range(self._type_num):
                    previous_num = 0
                    for type in range(node_type):
                        previous_num += self._num_node_type[type]
                    batch_data_neg_list[event][neg][node_type] =\
                        [neg_node+previous_num for neg_node in batch_data_neg_list[event][neg][node_type]]
                    # neg_event_partition[neg][batch_data_neg_list[event][neg][node_type]] = node_type+1
                    batch_feed_dict[self.placeholders['negevents_nodes_type_ph'][event][neg][node_type]] = \
                        batch_data_neg_list[event][neg][node_type]
                # batch_feed_dict[self.placeholders['neg_event_partition_idx_ph'][event][neg]] = neg_event_partition[neg]
        return batch_feed_dict, epoch_flag


    def get_feed_dict_eval(self):
        feed_dict_eval = {}
        eval_data, epoch_flag = self.test_data.next_batch()
        neg_num = 1
        eval_data_neg = self.sample_neg_events(eval_data, neg_num)
        eval_data = eval_data[0]
        eval_data_neg = eval_data_neg[0][0]
        feed_dict_eval[self.placeholders['node_embedding_ph']] = self.node_embedding_cur
        feed_dict_eval[self.placeholders['node_cellstates_ph']] = self.node_cellstates_cur
        ###################feed neighbor#####################
        node_list = []
        for node_type in range(self._type_num):
            previous_num = 0
            for type in range(node_type):
                previous_num += self.params['n_nodes_pertype'][type]
            eval_data[node_type] = [node+previous_num for node in eval_data[node_type]]
            node_list += eval_data[node_type]
        for nodei in node_list:
            for nodej in node_list:
                if nodei != nodej:
                    self.cur_adj[nodei][nodej] = 0
        neighbor_nodes = []
        for node in node_list:
            neighbor_nodes.extend(list(np.where(self.cur_adj[node] > 0)[0]))
        if len(neighbor_nodes) == 0:
            feed_dict_eval[self.placeholders['is_neighbor_eval']] = False
            neighbor_nodes = node_list
        else:
            feed_dict_eval[self.placeholders['is_neighbor_eval']] = True
            neighbor_nodes = sorted(np.unique(neighbor_nodes))
        ############neighbor link type########
        # neighbor_type = [[0]*self._type_num]*len(neighbor_nodes)
        # for index in range(len(neighbor_nodes)):
        #     for node_type in range(self._type_num):
        #         neighbor_type[index][node_type] = sum(self.cur_adj[neighbor_nodes[index]][eval_data[node_type]])
        # feed_dict_eval[self.placeholders['neighbor_type_eval']] = np.asarray(neighbor_type, dtype=np.int32)
        ###############feed_ajd#######################
        # neighbors = self.cur_adj[np.ix_(neighbor_nodes, node_list)]
        # neighbor_indicese, neighbor_values, neighbor_shape = self.sparse_to_tuple(neighbors)
        # feed_dict_eval[self.placeholders['cur_adj_eval_ph']] = (neighbor_indicese, neighbor_values, neighbor_shape)
        ###########################################
        for nodei in node_list:
            for nodej in node_list:
                if nodei != nodej:
                    self.cur_adj[nodei][nodej] = 1
        neighbor_partition = np.zeros(self._num_node)
        neighbor_partition[neighbor_nodes] = 1
        feed_dict_eval[self.placeholders['neighbor_partition_idx_eval_ph']] = neighbor_partition
        ##################feed event time############
        feed_dict_eval[self.placeholders['nodes_last_time_ph']] = self.node_last_time
        feed_dict_eval[self.placeholders['events_time_eval_ph']] = eval_data[-1]
        #############################################
        event_partition = np.zeros(self._num_node, dtype=np.int32)
        # neg_event_partition = np.zeros(self._num_node, dtype=np.int32)
        for node_type in range(self._type_num):
            previous_num = 0
            for type in range(node_type):
                previous_num += self._num_node_type[type]
            eval_data_neg[node_type] = [neg_node + previous_num for neg_node in eval_data_neg[node_type]]
            feed_dict_eval[self.placeholders['events_nodes_type_eval_ph'][node_type]] =\
                    np.asarray(eval_data[node_type], dtype=np.int32)
            event_partition[eval_data[node_type]] = node_type+1
            feed_dict_eval[self.placeholders['negevents_nodes_type_eval_ph'][node_type]] = eval_data_neg[node_type]
            # neg_event_partition[eval_data_neg[node_type]] = node_type+1
        feed_dict_eval[self.placeholders['event_partition_idx_eval_ph']] = event_partition
        # feed_dict_eval[self.placeholders['neg_event_partition_idx_eval_ph']] = neg_event_partition
        # feed_dict_eval[self.placeholders['event_stitch_idx_eval_ph'][0]] = np.where(event_partition==0)[0].tolist()
        return feed_dict_eval, epoch_flag








