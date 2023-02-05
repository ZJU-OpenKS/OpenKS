import sys
import random
import numpy as np
import tensorflow as tf
from utils.inits import glorot_init
import sklearn.metrics as metrics
from models.basic_model import BasicModel
from layers.layers import Dense, TimeDymLayer
from utils.data_manager import *
import scipy.sparse as sp
# from scipy.sparse import linalg
import time
import pickle as pkl

class GnnEventModel_changed(BasicModel):
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
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'changed'
        return log_file

    def get_checkpoint_dir(self, params):
        checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'changed'
        return checkpoint_dir

    def make_model(self):
        #compute loss
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['loss'] = self.build_specific_graph_model()
            self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict'], self.ops['node_vec'], \
                self.ops['node_cel'] = self.build_specific_eval_graph_model()
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    def _create_placeholders(self):
            self.placeholders['events_nodes_type_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                        name='node_type%i_event%i_ph'%(node_type, event))
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

            self.placeholders['node_embedding_ph'] = [tf.placeholder(tf.float64,
                                                    shape=[self._num_node_type[node_type], self._h_dim],
                                                    name='node_type%i_embedding_ph' % node_type)
                                                    for node_type in range(self._type_num)]
            # self.placeholders['node_embedding_ph'] = tf.placeholder(tf.float64,
            #                                         shape=[self._num_node, self._h_dim], name='node_embedding_ph')

            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self.placeholders['node_cellstates_ph'] = [tf.placeholder(tf.float64,
                                                    shape=[self._num_node_type[node_type], self._h_dim],
                                                    name='node_type%i_cellstates_ph' % node_type)
                                                    for node_type in range(self._type_num)]
                # self.placeholders['node_cellstates_ph'] = tf.placeholder(tf.float64,
                #                                         shape=[self._num_node, self._h_dim], name='node_cellstates_ph')

            self.placeholders['negevents_nodes_type_ph'] = [[[tf.placeholder(tf.int32, shape=[None],
                                                    name='node_type%i_event%i_neg%i_ph' % (node_type, event, neg))
                                                    for node_type in range(self._type_num)]
                                                    for event in range(self._eventnum_batch)]
                                                    for neg in range(self._neg_num)]

            self.placeholders['is_train'] = tf.placeholder(tf.bool, name='is_train')

            # self._node_lasttime = [tf.get_variable('node_type%i_last_time' % (node_type),
            #                                        shape=[self._num_node_type[node_type], 1],
            #                                        dtype=tf.float64, trainable=False,
            #                                        initializer=tf.zeros_initializer())
            #                        for node_type in range(self._type_num)]

            # self.placeholders['nodes_type_last_time_ph'] = [tf.placeholder(tf.float64,
            #                                             shape=[self._num_node_type[node_type]],
            #                                             name='node_type%i_last_time_ph' % node_type)
            #                                             for node_type in range(self._type_num)]

            # self.placeholders['events_time_ph'] = [tf.placeholder(tf.float64,
            #                                         shape=[1], name='event%i_time_ph' % event)
            #                                         for event in range(self._eventnum_batch)]
            #
            # self.placeholders['events_time_eval_ph'] = tf.placeholder(tf.float64, shape=[1])

            # self.placeholders['cur_adj_ph'] = [tf.sparse.placeholder(tf.float64, shape=[self._num_node, self._num_node])
            #                                 for event in range(self._eventnum_batch)]

###########test placeholder###########################
            self.placeholders['events_nodes_type_eval_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                            name='node_type%i_eval_ph' % (node_type))
                                                            for node_type in range(self._type_num)]

            self.placeholders['event_partition_idx_eval_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                            name='event_partition_idx_type%i_eval_ph' % (node_type))
                                                            for node_type in range(self._type_num)]

            self.placeholders['event_stitch_idx_eval_ph'] = [[
                tf.placeholder(tf.int32, shape=[None]),
                tf.placeholder(tf.int32, shape=[None])
            ] for node_type in range(self._type_num)]

            self.placeholders['negevents_nodes_type_eval_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                            name='node_type%i_eval_ph' % (node_type))
                                                            for node_type in range(self._type_num)]

            # self.placeholders['cur_adj_eval_ph'] = tf.sparse.placeholder(tf.float64, shape=[self._num_node, self._num_node])

######################################################

    def _create_variables(self):
        cur_seed = random.getrandbits(32)

        # self._node_embedding_init = tf.get_variable('node_embedding_init', shape=[self._num_node, self._h_dim],
        #                             dtype=tf.float64, trainable=True,
        #                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        #
        # self._node_embedding_st = tf.get_variable('node_embedding_st', shape=[self._num_node, self._h_dim],
        #                             dtype=tf.float64, trainable=True,
        #                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        #
        # if self.params['graph_rnn_cell'].lower() == 'lstm':
        #     self._node_cellstates_init = tf.get_variable('node_cellstates_init', shape=[self._num_node, self._h_dim],
        #                             dtype=tf.float64, trainable=False, initializer=tf.zeros_initializer())

        # self.diffusion_weight = tf.get_variable('diffusion_weight', [self._h_dim * self._max_diffusion_step, self._h_dim], dtype=tf.float64,
        #                          initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        # self.diffusion_bias = tf.get_variable('diffusion_biases', [self._h_dim], dtype=tf.float64,
        #                        initializer=tf.zeros_initializer())

        self._node_embedding_init = [tf.get_variable('node_type%i_embedding_init'%(node_type),
                                shape=[self._num_node_type[node_type], self._h_dim],
                                dtype=tf.float64, trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]
        self._node_embedding_st = [tf.get_variable('node_type%i_embedding_st' % (node_type),
                                shape=[self._num_node_type[node_type], self._h_dim],
                                dtype=tf.float64, trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]

        if self.params['graph_rnn_cell'].lower() == 'lstm':
            self._node_cellstates_init = [tf.get_variable('node_type%i_cellstates_init'%(node_type),
                                    shape=[self._num_node_type[node_type], self._h_dim],
                                    dtype=tf.float64, trainable=False,
                                    initializer=tf.zeros_initializer())
                                    for node_type in range(self._type_num)]

    # def calculate_random_walk_matrix(self, adj_mx, type_w):
    #     d = tf.reduce_sum(adj_mx, 1)
    #     d_inv = tf.reciprocal(d)
    #     d_inv = tf.diag(d_inv)
    #     # adj_mx = tf.contrib.layers.dense_to_sparse(adj_mx)
    #     # random_walk_mx = tf.sparse_tensor_dense_matmul(adj_mx, d_inv)
    #     type_w = tf.diag(type_w)
    #     random_walk_w = tf.matmul(type_w, adj_mx)
    #     random_walk_mx = tf.matmul(d_inv, random_walk_w)
    #
    #
    #     # adj_mx = sp.coo_matrix(adj_mx)
    #     # d = np.array(adj_mx.sum(1))
    #     # d_inv = np.power(d, -1).flatten()
    #     # d_inv[np.isinf(d_inv)] = 0.
    #     # d_mat_inv = sp.diags(d_inv)
    #     # random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    #     return random_walk_mx
    #
    # def _build_sparse_matrix(self, L):
    #     # L = L.tocoo()
    #     # indices = np.column_stack((L.row, L.col))
    #     # L = tf.SparseTensor(indices, L.data, L.shape)
    #     # return tf.sparse_reorder(L)
    #     idx = tf.where(tf.not_equal(L, 0.0))
    #     sparse = tf.SparseTensor(idx, tf.gather_nd(L, idx), tf.shape(L, out_type=tf.int64))
    #     return tf.sparse_reorder(sparse)

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._neg_num = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']
        # self._num_node = sum(self._num_node_type)
        # self._max_diffusion_step = self.params['max_diffusion_step']
        # self._eventclass = self.hegraph.get_eventtype_num()
        self._keep = self.params['keep_prob']
        self._istraining = self.params['is_training']

        self._create_placeholders()
        self._create_variables()

        # self.last_time = [np.zeros(self._num_node_type[node_type])
        #                   for node_type in range(self._type_num)]

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            self.activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
           self.activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['type_weights'] = [tf.Variable(glorot_init([self._h_dim, self._h_dim],
                                        name='gnn_type_weights_typ%i' % node_type))
                                        for node_type in range(self._type_num)]

        self.weights['type_weights_scalar'] = [tf.Variable(1.0, dtype=tf.float64, trainable=True) for node_type in range(self._type_num)]
        # self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._eventclass]))
        self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._h_dim]))

        cell_type = self.params['graph_rnn_cell'].lower()
        self.weights['rnn_cells'] = {}
        if self.params['use_different_cell']:
            for node_type in range(self._type_num):
                if cell_type == 'lstm':
                    cell = tf.nn.rnn_cell.BasicLSTMCell(self._h_dim, activation=self.activation_fun)
                elif cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(self._h_dim, activation=self.activation_fun)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(self._h_dim, activation=self.activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.params['keep_prob'])
                self.weights['rnn_cells'][node_type] = cell
        else:
            if cell_type == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell(self._h_dim, activation=self.activation_fun)
            elif cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(self._h_dim, activation=self.activation_fun)
            elif cell_type == 'rnn':
                cell = tf.nn.rnn_cell.BasicRNNCell(self._h_dim, activation=self.activation_fun)
            else:
                raise Exception("Unknown RNN cell type '%s'." % cell_type)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self._keep if self._istraining else 1.)
            for node_type in range(self._type_num):
                self.weights['rnn_cells'][node_type] = cell

        # self.tabel_z = [0 for _ in range(self._type_num)]
        # self.tabel_T = [[] for _ in range(self._type_num)]


    def build_specific_graph_model(self):
        node_vec = [[None] * self._type_num] * (self._eventnum_batch + 1)
        # node_vec = [None] * (self._eventnum_batch + 1)
        node_vec[0] = self._node_embedding_init
        # node_last_time = [[None]* self._type_num] * (self._eventnum_batch + 1)
        # node_last_time[0] = self.placeholders['nodes_type_last_time_ph']
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            node_cel = [[None] * self._type_num] * (self._eventnum_batch + 1)
            # node_cel = [None] * (self._eventnum_batch+1)
            node_cel[0] = self._node_cellstates_init
        event_states_list = []
        neg_event_states_list = []
        self.dense_layer = Dense(2 * self._h_dim, self._h_dim, act=lambda x: x, keep=self._keep if self._istraining else 1.)
        # self.timedyn_layer = TimeDymLayer(self._h_dim, keep=self._keep if self._istraining else 1.)
        for event_id in range(self._eventnum_batch):
            event_states = tf.zeros([self._h_dim], dtype=tf.float64)
            neg_event_states = [tf.zeros([self._h_dim], dtype=tf.float64) for _ in range(self._neg_num)]
            for node_type in range(self._type_num):
                states_dy = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                self.placeholders['events_nodes_type_ph'][event_id][node_type])
                states_st = tf.nn.embedding_lookup(self._node_embedding_st[node_type],
                                                   self.placeholders['events_nodes_type_ph'][event_id][node_type])
                ####################concat embedding################
                concate_states = tf.concat([states_st, states_dy], 1)
                states = self.dense_layer(concate_states)
                ####################embedding with gate#############
                # last_time = tf.expand_dims(tf.nn.embedding_lookup(node_last_time[event_id][node_type],
                #                                    self.placeholders['events_nodes_type_ph'][event_id][node_type]),1)
                # event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_ph'][event_id], 1),
                #                      [tf.shape(last_time)[0], 1])
                # delta_time = event_time - last_time
                # states = self.timedyn_layer([delta_time, states_st, states_dy])
                # event_types = self.timedyn_layer([delta_time, states_st, states_dy])
                #####################################################
                states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
                # event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])
                event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], states_aggregated)
            # event_states = self.activation_fun(tf.matmul(event_states, self.weights['event_weights'])) + event_states
            for neg in range(self._neg_num):
                for node_type in range(self._type_num):
                    neg_states_dy = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                self.placeholders['negevents_nodes_type_ph'][neg][event_id][node_type])
                    neg_states_st = tf.nn.embedding_lookup(self._node_embedding_st[node_type],
                                                self.placeholders['negevents_nodes_type_ph'][neg][event_id][node_type])
                    #################concat embedding##################
                    neg_concate_states = tf.concat([neg_states_st, neg_states_dy], 1)
                    neg_states = self.dense_layer(neg_concate_states)
                    #################embedding with gate###############
                    # neg_last_time = tf.expand_dims(tf.nn.embedding_lookup(node_last_time[event_id][node_type],
                    #                     self.placeholders['negevents_nodes_type_ph'][neg][event_id][node_type]),1)
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
            for node_type in range(self._type_num):
                node_vec_part = tf.dynamic_partition(node_vec[event_id][node_type],
                                                self.placeholders['event_partition_idx_ph'][event_id][node_type], 2)
                # node_last_time_part = tf.dynamic_partition(node_last_time[event_id][node_type],
                #                                 self.placeholders['event_partition_idx_ph'][event_id][node_type], 2)
                # target_embedding = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                #                                 self.placeholders['events_nodes_type_ph'][event_id][node_type])
                target_embedding = node_vec_part[1]
                sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
                # if self.params['graph_rnn_cell'].lower() == 'lstm':
                node_cel_part = tf.dynamic_partition(node_cel[event_id][node_type],
                                            self.placeholders['event_partition_idx_ph'][event_id][node_type],2)
                # target_celstates = tf.nn.embedding_lookup(node_cel[event_id][node_type],
                #                                     self.placeholders['events_nodes_type_ph'][event_id][node_type])
                target_celstates = node_cel_part[1]
                _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=sent_states,
                                                                    state=(target_celstates, target_embedding))
                new_target_celstates = new_target_states_ch[0]
                new_target_embedding = new_target_states_ch[1]

                node_cel[event_id + 1][node_type] = tf.dynamic_stitch(
                    self.placeholders['event_stitch_idx_ph'][event_id][node_type],
                    [node_cel_part[0], new_target_celstates])
                # else:
                #     new_target_embedding = self.weights['rnn_cells'][node_type](inputs=sent_states, state=target_embedding)[1]

                # new_last_time = tf.tile(self.placeholders['events_time_ph'][event_id],[tf.shape(target_embedding)[0]])
                # node_last_time[event_id + 1][node_type] = tf.dynamic_stitch(
                #     self.placeholders['event_stitch_idx_ph'][event_id][node_type],
                #     [node_last_time_part[0], new_last_time])
                #####states propagation#############
                # if self.params['graph_rnn_cell'].lower() == 'lstm':
                #     node_cel[event_id+1] = tf.add(node_cel[event_id+1],
                #                                 self.gconv(new_target_celstates, self.placeholders['cur_adj_ph'][event_id],
                #                                 self.placeholders['events_nodes_type_ph'][event_id][node_type]))
                # node_vec[event_id+1] = tf.tanh(node_cel[event_id+1])
                #####################################

                # node_vec_part = tf.dynamic_partition(node_vec[event_id+1],
                #                             self.placeholders['event_partition_idx_ph'][event_id][node_type], 2)

                node_vec[event_id + 1][node_type] = tf.dynamic_stitch(
                    self.placeholders['event_stitch_idx_ph'][event_id][node_type],
                    [node_vec_part[0], new_target_embedding])

        event_scores = tf.stack(event_states_list)
        neg_event_scores = tf.stack(neg_event_states_list)
        event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores), axis=1))
        neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores), axis=2))

        # predict = tf.tanh(event_scores_norms/2)
        # neg_predict = tf.tanh(neg_event_scores_norms/2)

        event_losses = tf.log(tf.tanh(event_scores_norms/2))
        neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms)/2))
        neg_event_losses = tf.reduce_sum(neg_event_losses, axis=1)
        # neg_event_losses = tf.reduce_sum(tf.multiply(neg_predict, neg_event_losses), axis=1)
        losses = event_losses + neg_event_losses
        # losses = event_losses
        loss_mean = -tf.reduce_mean(losses)
        # rmse = tf.reduce_mean(tf.square(tf.subtract(predict, tf.ones([self._eventnum_batch],dtype=tf.float64))))
        # negrmse = tf.reduce_mean(tf.square(tf.subtract(neg_predict, tf.zeros([self._eventnum_batch, self._neg_num],dtype=tf.float64))))
        return loss_mean

    def build_specific_eval_graph_model(self):
        node_vec = [None] * self._type_num
        # node_last_time = [None]
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            node_cel = [None] * self._type_num
        event_states = tf.zeros([self._h_dim], dtype=tf.float64)
        neg_event_states = tf.zeros([self._h_dim], dtype=tf.float64)
        for node_type in range(self._type_num):
            states_dy = tf.nn.embedding_lookup(self.placeholders['node_embedding_ph'][node_type],
                                               self.placeholders['events_nodes_type_eval_ph'][node_type])
            states_st = tf.nn.embedding_lookup(self._node_embedding_st[node_type],
                                               self.placeholders['events_nodes_type_eval_ph'][node_type])
            ##########concat embedding##########
            concate_states = tf.concat([states_st, states_dy], 1)
            states = self.dense_layer(concate_states)
            ##########embedding with gate#######
            # last_time = tf.expand_dims(tf.nn.embedding_lookup(self.placeholders['nodes_type_last_time_ph'][node_type],
            #                                     self.placeholders['events_nodes_type_eval_ph'][node_type]), 1)
            # event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_eval_ph'], 1),
            #                      [tf.shape(last_time)[0], 1])
            # delta_time = event_time - last_time
            # states = self.timedyn_layer([delta_time, states_st, states_dy])
            # event_types = self.timedyn_layer([delta_time, states_st, states_dy])
            ####################################
            states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
            # event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])
            event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], states_aggregated)
            # event_states = self.activation_fun(tf.matmul(event_states, self.weights['event_weights'])) + event_states
            neg_states_dy = tf.nn.embedding_lookup(self.placeholders['node_embedding_ph'][node_type],
                                                   self.placeholders['negevents_nodes_type_eval_ph'][node_type])
            neg_states_st = tf.nn.embedding_lookup(self._node_embedding_st[node_type],
                                                   self.placeholders['negevents_nodes_type_eval_ph'][node_type])
            #############concat negembedding##############
            neg_concate_states = tf.concat([neg_states_st, neg_states_dy], 1)
            neg_states = self.dense_layer(neg_concate_states)
            #############embedding with gate#############
            # neg_last_time = tf.expand_dims(tf.nn.embedding_lookup(self.placeholders['nodes_type_last_time_ph'][node_type],
            #                                     self.placeholders['negevents_nodes_type_eval_ph'][node_type]), 1)
            # neg_event_time = tf.tile(tf.expand_dims(self.placeholders['events_time_eval_ph'], 1),
            #                      [tf.shape(neg_last_time)[0], 1])
            # neg_delta_time = neg_event_time - neg_last_time
            # neg_states = self.timedyn_layer([neg_delta_time, neg_states_st, neg_states_dy])
            ###############################################
            neg_states_aggregated = tf.reduce_mean(neg_states, axis=0, keepdims=True)
            # neg_event_states[neg] += tf.matmul(neg_states_aggregated, self.weights['type_weights'][node_type])
            neg_event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], neg_states_aggregated)
                # neg_event_states[neg] = self.activation_fun(tf.matmul(neg_event_states[neg], self.weights['event_weights'])) + neg_event_states[neg]
        for node_type in range(self._type_num):
            node_vec_part = tf.dynamic_partition(self.placeholders['node_embedding_ph'][node_type],
                                                 self.placeholders['event_partition_idx_eval_ph'][node_type], 2)
            # node_last_time_part = tf.dynamic_partition(self.placeholders['nodes_type_last_time_ph'][node_type],
            #                                     self.placeholders['event_partition_idx_eval_ph'][node_type], 2)
            target_embedding = node_vec_part[1]
            sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
            # if self.params['graph_rnn_cell'].lower() == 'lstm':
            node_cel_part = tf.dynamic_partition(self.placeholders['node_cellstates_ph'][node_type],
                                                 self.placeholders['event_partition_idx_eval_ph'][node_type], 2)
            target_celstates = node_cel_part[1]
            _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=sent_states,
                                                                    state=(target_celstates, target_embedding))
            new_target_celstates = new_target_states_ch[0]
            new_target_embedding = new_target_states_ch[1]

            node_cel[node_type] = tf.dynamic_stitch(self.placeholders['event_stitch_idx_eval_ph'][node_type],
                                                                [node_cel_part[0], new_target_celstates])
            # else:
            #     new_target_embedding = \
            #     self.weights['rnn_cells'][node_type](inputs=sent_states, state=target_embedding)[1]

            #####states propagation#############
            # if self.params['graph_rnn_cell'].lower() == 'lstm':
            #     node_cel = tf.add(node_cel, self.gconv(new_target_celstates,
            #                                     self.placeholders['cur_adj_eval_ph'],
            #                                     self.placeholders['events_nodes_type_eval_ph'][node_type]))
            # node_vec = tf.tanh(node_cel)
            ####################################

            # node_vec_part = tf.dynamic_partition(node_vec,
            #                                      self.placeholders['event_partition_idx_eval_ph'][node_type], 2)

            node_vec[node_type] = tf.dynamic_stitch(self.placeholders['event_stitch_idx_eval_ph'][node_type],
                                         [node_vec_part[0], new_target_embedding])
            # new_last_time = tf.tile(self.placeholders['events_time_eval_ph'], [tf.shape(target_embedding)[0]])
            # node_last_time[node_type] = tf.dynamic_stitch(
            #     self.placeholders['event_stitch_idx_eval_ph'][node_type],
            #     [node_last_time_part[0], new_last_time])

        event_scores = tf.squeeze(event_states)
        neg_event_scores = tf.squeeze(neg_event_states)
        event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores)))
        neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores)))

        predict = tf.tanh(event_scores_norms / 2)
        neg_predict = tf.tanh(neg_event_scores_norms / 2)

        # predict = tf.tanh(event_scores_norms/2)
        # neg_predict = tf.tanh(neg_event_scores_norms/2)
        event_losses = tf.log(tf.tanh(event_scores_norms / 2))
        neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms) / 2))
        loss_mean = -(event_losses + neg_event_losses)
        # test_loss = tf.log(predict)
        # test_loss1 = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms)/2))
        # rmse = tf.reduce_mean(tf.square(tf.subtract(predict, tf.ones([self._eventnum_batch],dtype=tf.float64))))
        # negrmse = tf.reduce_mean(tf.square(tf.subtract(neg_predict, tf.zeros([self._eventnum_batch, self._neg_num],dtype=tf.float64))))
        return loss_mean, predict, neg_predict, node_vec, node_cel

    # def _concat(self, x, x_):
    #     x_ = tf.expand_dims(x_, 0)
    #     return tf.concat([x, x_], axis=0)
    #
    # def gconv(self, inputs, adj_mx, start_nodes):
    #     # x0 = tf.nn.embedding_lookup(inputs, start_nodes)
    #     x0 = inputs
    #     type_ws = [tf.tile([self.weights['type_weights_scalar'][node_type]], [self._num_node_type[node_type]])
    #               for node_type in range(self._type_num)]
    #     type_w = tf.concat(type_ws, 0)
    #     # x0_to = tf.nn.embedding_lookup(inputs, to_nodes)
    #     adj_mx0 = tf.nn.embedding_lookup(adj_mx, start_nodes)
    #     support = self.calculate_random_walk_matrix(adj_mx, type_w)
    #     type_w0 = tf.nn.embedding_lookup(type_w, start_nodes)
    #     support0 = self.calculate_random_walk_matrix(adj_mx0, type_w0)
    #     support = self._build_sparse_matrix(support)
    #     support0 = self._build_sparse_matrix(support0)
    #     # x = tf.expand_dims(x0, axis=0)
    #     x1 = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(support0), x0)
    #     # x = self._concat(x, x1)
    #     x = tf.expand_dims(x1, 0)
    #     for k in range(2, self._max_diffusion_step+1):
    #         x2 = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(support), x1)
    #         x = self._concat(x, x2)
    #         x1 = x2
    #     # num_matrics = self._max_diffusion_step
    #     x = tf.reshape(x, shape=[self._max_diffusion_step, self._num_node, self._h_dim])
    #     x = tf.transpose(x, perm=[1, 2, 0])
    #     x = tf.reshape(x, shape=[self._num_node, self._h_dim*self._max_diffusion_step])
    #     x = tf.add(tf.matmul(x, self.diffusion_weight), self.diffusion_bias)
    #     return x

    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        print('batches'+str(train_batches_num))
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []
                # self.last_time = [np.zeros(self._num_node_type[node_type]) for node_type in range(self._type_num)]
                # self.cur_adj = np.zeros([self._num_node, self._num_node])
                # cost_test_list_train = []
                # cost_test_list1_train = []
                epoch_flag = False
                print('start epoch %i'%(epoch))
                while not epoch_flag:
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict()
                    fetches = [self.ops['loss'], self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                    cost, step, lr, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)
                    epoch_loss.append(cost)
                    # cost_test_list_train.append(cost_test)
                    # cost_test_list1_train.append(cost_test1)
                    if np.isnan(cost):
                        log_out.write('Train ' + str(epoch) + ':Nan error!')
                        print('Train ' + str(epoch) + ':Nan error!')
                        return
                    if step == 1 or step % (self.params['decay_step']/10) == 0:
                        avgc = np.mean(epoch_loss)
                        log_out.write('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        print(cost)
                        # print('cost test' + str(np.mean(cost_test_list_train)))
                        # print('cost test1' + str(np.mean(cost_test_list1_train)))
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        sys.stdout.flush()
                    # if step == 1:
                    #     valid_loss = self.validation()
                    #     print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        # self.test()
                        # self.test()
                        # sys.stdout.flush()
                # if step == 1 or step % self.params['eval_point'] == 0:
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

    # def validation(self):
    #     valid_batches_num = self.valid_data.get_batch_num()
    #     valid_loss = []
    #     epoch_flag = False
    #     # cost_test_list = []
    #     # cost_test_list1 = []
    #     print('valid batches' + str(valid_batches_num))
    #     while not epoch_flag:
    #         fetches = self.ops['loss']
    #         batch_feed_dict, epoch_flag = self.get_batch_feed_dict('valid')
    #         cost = self.sess.run(fetches, feed_dict=batch_feed_dict)
    #         if np.isnan(cost):
    #             print('Evaluation loss Nan!')
    #             sys.exit(1)
    #         valid_loss.append(cost)
    #         # cost_test_list.append(cost_test)
    #         # cost_test_list1.append(cost_test1)
    #     # print('valid cost test'+str(np.mean(cost_test_list)))
    #     # print('valid cost test'+str(np.mean(cost_test_list1)))
    #     return np.mean(valid_loss)

    def validation(self):
        valid_loss = []
        self.valid_data.batch_size = 1
        valid_batches_num = self.valid_data.get_batch_num()
        print('valid nums:' + str(valid_batches_num))
        epoch_flag = False
        self.node_embedding_evalcur = [None for _ in range(self._type_num)]
        self.node_cellstates_evalcur = [None for _ in range(self._type_num)]
        for node_type in range(self._type_num):
            self.node_embedding_evalcur[node_type] = self._node_embedding_init[node_type].eval(session=self.sess)
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self.node_cellstates_evalcur[node_type] = self._node_cellstates_init[node_type].eval(session=self.sess)
        while not epoch_flag:
            fetches = [self.ops['loss_eval'], self.ops['node_vec'], self.ops['node_cel']]
            feed_dict_valid, epoch_flag = self.get_feed_dict_eval('valid')
            cost, self.node_embedding_evalcur, self.node_cellstates_evalcur = self.sess.run(fetches, feed_dict=feed_dict_valid)
            valid_loss.append(cost)
        return np.mean(valid_loss)


    def test(self):
        self.test_data.batch_size = 1
        test_batches_num = self.test_data.get_batch_num()
        print('test nums:'+str(test_batches_num))
        epoch_flag = False
        val_preds = []
        labels = []
        # cost_test_list = []
        # cost_test_list1 = []
        self.node_embedding_evalcur = [None for _ in range(self._type_num)]
        self.node_cellstates_evalcur = [None for _ in range(self._type_num)]
        for node_type in range(self._type_num):
            self.node_embedding_evalcur[node_type] = self._node_embedding_init[node_type].eval(session=self.sess)
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self.node_cellstates_evalcur[node_type] = self._node_cellstates_init[node_type].eval(session=self.sess)
        # with open('/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/test_neg.pkl', 'rb') as test_neg_root:
        #     test_data_neg_list = pkl.load(test_neg_root)
        # num = 0
        while not epoch_flag:
            # test_data_neg = test_data_neg_list[num]
            fetches = [self.ops['predict'], self.ops['neg_predict'], self.ops['node_vec'], self.ops['node_cel']]
            feed_dict_test, epoch_flag = self.get_feed_dict_eval('test')
            predict, neg_predict, self.node_embedding_evalcur, self.node_cellstates_evalcur = \
                self.sess.run(fetches, feed_dict=feed_dict_test)
            val_preds.append(predict)
            val_preds.append(neg_predict)
            labels.append(1)
            labels.append(0)
            # cost_test_list.append(cost_test)
            # cost_test_list1.append(cost_test1)
        # precision = metrics.precision_score(labels, val_preds, average=None)
        # recall = metrics.recall_score(labels, val_preds, average=None)
        # f1 = metrics.f1_score(labels, val_preds, average=None)
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        # print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)
        print('mae:%f, rmse:%f' % (mae, rmse))
        # print('test cost test' + str(np.mean(cost_test_list)))
        # print('test cost test' + str(np.mean(cost_test_list1)))


    def sample_negbatch_events(self, batch_data):
        #     batch_data_neg_list = []
        #     for neg in range(self._neg_num):
        #         batch_data_neg = [[[] for _ in range(self._type_num)] for _ in range(self._eventnum_batch)]
        #         for event in range(self._eventnum_batch):
        #             for type in range(self._type_num):
        #                 tabel_size = len(tabel_T[type])
        #                 while(len(batch_data_neg[event][type])<len(batch_data[event][type])):
        #                     neg_node = tabel_T[type][random.randint(0, tabel_size - 1)]
        #                     if (neg_node in batch_data[event][type]) or (neg_node in batch_data_neg[event][type]):
        #                         continue
        #                     batch_data_neg[event][type].append(neg_node)
        #         batch_data_neg_list.append(batch_data_neg)
        #     return batch_data_neg_list
        batch_data_neg_list = []
        for neg in range(self._neg_num):
            batch_data_neg = [[[] for _ in range(self._type_num)] for _ in range(self._eventnum_batch)]
            for event in range(self._eventnum_batch):
                for type in range(self._type_num):
                    while (len(batch_data_neg[event][type]) < len(batch_data[event][type])):
                        neg_node = random.randint(0, self._num_node_type[type]-1)
                        if neg_node in batch_data_neg[event][type]:
                            continue
                        batch_data_neg[event][type].append(neg_node)
            batch_data_neg_list.append(batch_data_neg)
        return batch_data_neg_list

    def sample_negeval_events(self, test_data):
        test_data_neg = [[] for _ in range(self._type_num)]
        for type in range(self._type_num):
            while (len(test_data_neg[type]) < len(test_data[type])):
                neg_node = random.randint(0, self._num_node_type[type]-1)
                if neg_node in test_data_neg[type]:
                    continue
                test_data_neg[type].append(neg_node)
        return test_data_neg

    def get_batch_feed_dict(self):
        batch_feed_dict = {}
        # if state == 'train':
        batch_data, epoch_flag = self.train_data.next_batch()
        # elif state == 'valid':
        #     batch_data, epoch_flag = self.valid_data.next_batch()
        # else:
        #     print('state wrong')
        # self.tabel_z, self.tabel_T = self.gen_sampling_table_pertype(self.tabel_z, self.tabel_T, batch_data)
        # batch_data_neg_list = self.sample_batch_events(batch_data, self.tabel_T)
        batch_data_neg_list = self.sample_negbatch_events(batch_data)
        for event in range(self._eventnum_batch):
            # batch_feed_dict[self.placeholders['events_time_ph'][event]] = batch_data[event][-1]

            # node_list = []
            # for node_type in range(self._type_num):
            #     node_list += batch_data[event][node_type]
            # for nodei in node_list:
            #     for nodej in node_list:
            #         if nodei != nodej:
            #             self.cur_adj[nodei][nodej] = 1
            # batch_feed_dict[self.placeholders['cur_adj_ph'][event]] = self.cur_adj
            # batch_feed_dict[self.placeholders['cur_adj_ph'][event]] =

            for node_type in range(self._type_num):
                event_partition = np.zeros(self._num_node_type[node_type])
                # event_partition = np.zeros(self._num_node)
                # previous_num = 0
                # for type in range(node_type):
                #     previous_num += self.params['n_nodes_pertype'][type]
                # batch_data[event][node_type] = [node+previous_num for node in batch_data[event][node_type]]
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(batch_data[event][node_type], dtype=np.int32)
                # batch_feed_dict[self.placeholders['nodes_type_last_time_ph'][node_type]] = self.last_time[node_type]
                event_partition[batch_data[event][node_type]] = 1
                batch_feed_dict[self.placeholders['event_partition_idx_ph'][event][node_type]] = event_partition
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][0]] = np.where(event_partition==0)[0].tolist()
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][1]] = np.where(event_partition==1)[0].tolist()
                # batch_feed_dict[self.placeholders['event_labels']] = np.asarray(batch_label, dtype=np.int32)
                for neg in range(self.params['negative_ratio']):
                    # batch_data_neg_list[neg][event][node_type] = [neg_node+previous_num for neg_node in batch_data_neg_list[neg][event][node_type]]
                    batch_feed_dict[self.placeholders['negevents_nodes_type_ph'][neg][event][node_type]] = \
                        np.asarray(batch_data_neg_list[neg][event][node_type], dtype=np.int32)
        return batch_feed_dict, epoch_flag

    # def get_feed_dict_test(self):
    #     feed_dict_test = {}
    #     test_data, epoch_flag = self.test_data.next_batch()
    #     test_data = test_data[0]
    #     for node_type in range(self._type_num):
    #         feed_dict_test[self.placeholders['node_embedding_ph'][node_type]] = self.node_embedding_testcur[node_type]
    #         if self.params['graph_rnn_cell'].lower() == 'lstm':
    #             feed_dict_test[self.placeholders['node_cellstates_ph'][node_type]] = self.node_cellstates_testcur[node_type]
    #     test_data_neg = self.sample_negtest_events(test_data)
    #     for node_type in range(self._type_num):
    #         event_partition = np.zeros(self._num_node_type[node_type])
    #         feed_dict_test[self.placeholders['events_nodes_type_eval_ph'][node_type]] = np.asarray(test_data[node_type], dtype=np.int32)
    #         event_partition[test_data[node_type]] = 1
    #         feed_dict_test[self.placeholders['event_partition_idx_eval_ph'][node_type]] = event_partition
    #         feed_dict_test[self.placeholders['event_stitch_idx_eval_ph'][node_type][0]] = \
    #         np.where(event_partition == 0)[0].tolist()
    #         feed_dict_test[self.placeholders['event_stitch_idx_eval_ph'][node_type][1]] = \
    #         np.where(event_partition == 1)[0].tolist()
    #         feed_dict_test[self.placeholders['negevents_nodes_type_eval_ph'][node_type]] = \
    #             np.asarray(test_data_neg[node_type], dtype=np.int32)
    #     return feed_dict_test, epoch_flag
    #
    # def get_feed_dict_valid(self):
    #     feed_dict_valid = {}
    #     valid_data, epoch_flag = self.valid_data.next_batch()
    #     valid_data = valid_data[0]
    #     for node_type in range(self._type_num):
    #         feed_dict_valid[self.placeholders['node_embedding_ph'][node_type]] = self.node_embedding_validcur[node_type]
    #         if self.params['graph_rnn_cell'].lower() == 'lstm':
    #             feed_dict_valid[self.placeholders['node_cellstates_ph'][node_type]] = self.node_cellstates_validcur[node_type]
    #     valid_data_neg = self.sample_negtest_events(valid_data)
    #     for node_type in range(self._type_num):
    #         event_partition = np.zeros(self._num_node_type[node_type])
    #         feed_dict_valid[self.placeholders['events_nodes_type_eval_ph'][node_type]] = np.asarray(valid_data[node_type], dtype=np.int32)
    #         event_partition[valid_data[node_type]] = 1
    #         feed_dict_valid[self.placeholders['event_partition_idx_eval_ph'][node_type]] = event_partition
    #         feed_dict_valid[self.placeholders['event_stitch_idx_eval_ph'][node_type][0]] = \
    #             np.where(event_partition == 0)[0].tolist()
    #         feed_dict_valid[self.placeholders['event_stitch_idx_eval_ph'][node_type][1]] = \
    #             np.where(event_partition == 1)[0].tolist()
    #         feed_dict_valid[self.placeholders['negevents_nodes_type_eval_ph'][node_type]] = \
    #             np.asarray(valid_data_neg[node_type], dtype=np.int32)
    #     return feed_dict_valid, epoch_flag

    def get_feed_dict_eval(self, type):
        feed_dict_eval = {}
        if type == 'valid':
            eval_data, epoch_flag = self.valid_data.next_batch()
        else:
            eval_data, epoch_flag = self.test_data.next_batch()
        eval_data = eval_data[0]
        # feed_dict_eval[self.placeholders['events_time_eval_ph']] = eval_data[-1]

        # node_list = []
        # for node_type in range(self._type_num):
        #     node_list += eval_data[node_type]
        # for nodei in node_list:
        #     for nodej in node_list:
        #         if nodei != nodej:
        #             self.cur_adj[nodei][nodej] = 1
        # feed_dict_eval[self.placeholders['cur_adj_eval_ph']] = self.cur_adj

        for node_type in range(self._type_num):
            feed_dict_eval[self.placeholders['node_embedding_ph'][node_type]] = self.node_embedding_evalcur[node_type]
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                feed_dict_eval[self.placeholders['node_cellstates_ph'][node_type]] = self.node_cellstates_evalcur[node_type]
        eval_data_neg = self.sample_negeval_events(eval_data)
        for node_type in range(self._type_num):
            event_partition = np.zeros(self._num_node_type[node_type])
            # event_partition = np.zeros(self._num_node)
            # previous_num = 0
            # for type in range(node_type):
            #     previous_num += self.params['n_nodes_pertype'][type]
            # eval_data[node_type] = [node + previous_num for node in eval_data[node_type]]
            # eval_data_neg[node_type] = [neg_node + previous_num for neg_node in eval_data_neg[node_type]]

            feed_dict_eval[self.placeholders['events_nodes_type_eval_ph'][node_type]] = np.asarray(eval_data[node_type], dtype=np.int32)
            # feed_dict_eval[self.placeholders['nodes_type_last_time_ph'][node_type]] = self.last_time[node_type]
            event_partition[eval_data[node_type]] = 1
            feed_dict_eval[self.placeholders['event_partition_idx_eval_ph'][node_type]] = event_partition
            feed_dict_eval[self.placeholders['event_stitch_idx_eval_ph'][node_type][0]] = \
                np.where(event_partition == 0)[0].tolist()
            feed_dict_eval[self.placeholders['event_stitch_idx_eval_ph'][node_type][1]] = \
                np.where(event_partition == 1)[0].tolist()
            feed_dict_eval[self.placeholders['negevents_nodes_type_eval_ph'][node_type]] = \
                np.asarray(eval_data_neg[node_type], dtype=np.int32)
        return feed_dict_eval, epoch_flag

    # def gen_sampling_table_pertype(self, tabel_z, tabel_T, batch_data):
    #     tabel_size = self.params['table_size']
    #     power = self.params['neg_power']
    #     nodes_degree_last, nodes_degree_cur = self.hegraph.get_curdegree_pertype(batch_data)
    #     for event in range(self._eventnum_batch):
    #         for type in range(self._type_num):
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
    #     return tabel_z, tabel_T

    # def sample_batch_events(self, batch_data, tabel_T):
    #     batch_data_neg_list = []
    #     for neg in range(self._neg_num):
    #         batch_data_neg = [[[] for _ in range(self._type_num)] for _ in range(self._eventnum_batch)]
    #         for event in range(self._eventnum_batch):
    #             for type in range(self._type_num):
    #                 tabel_size = len(tabel_T[type])
    #                 while(len(batch_data_neg[event][type])<len(batch_data[event][type])):
    #                     neg_node = tabel_T[type][random.randint(0, tabel_size - 1)]
    #                     if (neg_node in batch_data[event][type]) or (neg_node in batch_data_neg[event][type]):
    #                         continue
    #                     batch_data_neg[event][type].append(neg_node)
    #         batch_data_neg_list.append(batch_data_neg)
    #     return batch_data_neg_list

    # def get_batch_feed_dict_test(self, batch_data, node_embedding, node_cellstates):
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
    #         batch_feed_dict[self.placeholders['node_embedding_eval_ph'][node_type]] = node_embedding[node_type]
    #         batch_feed_dict[self.placeholders['node_cellstates_eval_ph'][node_type]] = node_cellstates[node_type]
    #     return batch_feed_dict







