import random
import numpy as np
import tensorflow as tf
from utils.inits import glorot_init, zeros_init
from utils.data_manager import BatchData
import sklearn.metrics as metrics
from models.basic_model import BasicModel

class GnnEventModel(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'use_event_bias': True
        })
        return params

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

            self.placeholders['node_embedding_eval_ph'] = [tf.placeholder(tf.float64,
                                                    shape=[self._num_node_type[node_type], self._h_dim],
                                                    name='node_type%i_embedding_ph' % node_type)
                                                    for node_type in range(self._type_num)]

            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self.placeholders['node_cellstates_eval_ph'] = [tf.placeholder(tf.float64,
                                                        shape=[self._num_node_type[node_type], self._h_dim],
                                                        name='node_type%i_cellstates_ph' % node_type)
                                                        for node_type in range(self._type_num)]

            self.placeholders['event_labels'] = tf.placeholder(tf.int32, shape=[self._eventnum_batch])

    def _create_variables(self):
        cur_seed = random.getrandbits(32)
        self._node_embedding = [tf.get_variable('node_type%i_embedding'%(node_type),
                                shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]
        # self._node_embedding_eval = [tf.get_variable('node_type%i_embedding_eval' % (node_type),
        #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
        #                         initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=cur_seed))
        #                         for node_type in range(self._type_num)]
        # self._node_cellstates = [tf.get_variable('node_type%i_cellstates'%(node_type),
        #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
        #                         initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        #                         for node_type in range(self._type_num)]
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            self._node_cellstates = [tf.get_variable('node_type%i_cellstates'%(node_type),
                                    shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=False,
                                    initializer=tf.zeros_initializer())
                                    for node_type in range(self._type_num)]
            # self._node_cellstates_eval = [tf.get_variable('node_type%i_cellstates_eval' % (node_type),
            #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64,
            #                         trainable=False, initializer=tf.zeros_initializer())
            #                         for node_type in range(self._type_num)]

        # self._node_embedding_eval = [tf.get_variable('node_type%i_embedding' % (node_type),
        #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=False,
        #                         initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        #                         for node_type in range(self._type_num)]
        #
        # self._node_cellstates_eval = [tf.get_variable('node_type%i_cellstates' % (node_type),
        #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=False,
        #                         initializer=tf.zeros_initializer())
        #                         for node_type in range(self._type_num)]

        self._assign_ops = []
        for node_type in range(self._type_num):
            self._assign_ops += [
            tf.assign(self._node_embedding[node_type], self.placeholders['node_embedding_eval_ph'][node_type])]
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self._assign_ops += [
                tf.assign(self._node_cellstates[node_type], self.placeholders['node_cellstates_eval_ph'][node_type])]
        self._assign_ops = tf.group(*self._assign_ops)

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._num_node_type = self.params['n_nodes_pertype']
        self._eventclass = self.hegraph.get_eventtype_num()

        self._create_placeholders()
        self._create_variables()

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            self.activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
           self. activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['type_weights'] = [tf.Variable(glorot_init([self._h_dim, self._h_dim],
                                        name='gnn_type_weights_typ%i' % node_type))
                                        for node_type in range(self._type_num)]
        # self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._eventclass]))
        self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._h_dim]))

        self.weights['predict_weights'] = tf.Variable(glorot_init([self._h_dim, 1]))

        if self.params['use_event_bias']:
            # self.weights['event_biases'] = tf.Variable(zeros_init([self._eventclass]))
            self.weights['predict_biases'] = tf.Variable(zeros_init([1]))

        cell_type = self.params['graph_rnn_cell'].lower()
        if cell_type == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell(self._h_dim, activation=self.activation_fun)
        elif cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(self._h_dim, activation=self.activation_fun)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(self._h_dim, activation=self.activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.params['keep_prob'])
        self.weights['rnn_cells'] = cell

    def build_specific_graph_model(self):
        node_vec = [[None] * self._type_num] * (self._eventnum_batch + 1)
        node_vec[0] = self._node_embedding
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            node_cel = [[None] * self._type_num] * (self._eventnum_batch + 1)
            node_cel[0] = self._node_cellstates
        event_states_list = []
        for event_id in range(self._eventnum_batch):
            event_states = tf.zeros([self._h_dim], dtype=tf.float64)
            for node_type in range(self._type_num):
                states = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                self.placeholders['events_nodes_type_ph'][event_id][node_type])
                states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
                event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])
                event_states = self.activation_fun(tf.matmul(event_states, self.weights['event_weights'])) + event_states
            event_states_list.append(tf.squeeze(event_states))

            for node_type in range(self._type_num):
                node_vec_part = tf.dynamic_partition(node_vec[event_id][node_type],
                                                self.placeholders['event_partition_idx_ph'][event_id][node_type], 2)
                target_embedding = tf.nn.embedding_lookup(node_vec[event_id][node_type],
                                                self.placeholders['events_nodes_type_ph'][event_id][node_type])
                sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
                if self.params['graph_rnn_cell'].lower() == 'lstm':
                    node_cel_part = tf.dynamic_partition(node_cel[event_id][node_type],
                                                         self.placeholders['event_partition_idx_ph'][event_id][node_type],2)
                    target_celstates = tf.nn.embedding_lookup(node_cel[event_id][node_type],
                                                        self.placeholders['events_nodes_type_ph'][event_id][node_type])
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

        event_scores = tf.stack(event_states_list)
        if self.params['use_event_bias']:
            event_scores = tf.matmul(event_scores, self.weights['predict_weights']) + self.weights['predict_biases']
        else:
            event_scores = tf.matmul(event_scores, self.weights['predict_weights'])
        # predict = tf.nn.softmax(event_scores)
        predict = tf.squeeze(event_scores)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['event_labels'], logits=event_scores)
        loss_mean = tf.losses.mean_squared_error(self.placeholders['event_labels'], predict)
        # loss_mean = tf.reduce_mean(loss)
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            return loss_mean, node_vec[event_id+1], node_cel[event_id+1], predict
        return loss_mean, node_vec[event_id+1], predict

    # def _create_placeholders_eval(self):
    #         self.placeholders['events_nodes_type_eval_ph'] = [tf.placeholder(tf.int32, [None],
    #                                                     name='node_type%i_eval_ph'%node_type)
    #                                                     for node_type in range(self._type_num)]
    #
    #         self.placeholders['event_partition_idx_eval_ph'] = [tf.placeholder(tf.int32, shape=[None],
    #                                                     name='event_partition_idx_type%i_eval_ph'%node_type)
    #                                                     for node_type in range(self._type_num)]
    #
    #         self.placeholders['event_stitch_idx_eval_ph'] = [[
    #             tf.placeholder(tf.int32, shape=[None]),
    #             tf.placeholder(tf.int32, shape=[None])
    #         ] for node_type in range(self._type_num)]
    #
    #         self.placeholders['node_embedding_eval_ph'] = [tf.placeholder(tf.float64,
    #                             shape=[self._num_node_type[node_type], self._h_dim],
    #                             name = 'node_type%i_embedding_ph' % node_type)
    #                             for node_type in range(self._type_num)]
    #
    #         self.placeholders['node_cellstates_eval_ph'] = [tf.placeholder(tf.float64,
    #                             shape=[self._num_node_type[node_type],self._h_dim],
    #                             name='node_type%i_cellstates_ph' % node_type)
    #                             for node_type in range(self._type_num)]


    # def _create_variables_eval(self):
    #     cur_seed = random.getrandbits(32)
    #     self._node_embedding_eval = [tf.get_variable('node_type%i_embedding'%(node_type),
    #                             shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=False,
    #                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
    #                             for node_type in range(self._type_num)]
    #     # self._node_cellstates = [tf.get_variable('node_type%i_cellstates'%(node_type),
    #     #                         shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
    #     #                         initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
    #     #                         for node_type in range(self._type_num)]
    #     self._node_cellstates_eval = [tf.get_variable('node_type%i_cellstates'%(node_type),
    #                             shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=False,
    #                             initializer=tf.zeros_initializer())
    #                             for node_type in range(self._type_num)]
    #
    #     self._assign_ops = []
    #     for node_type in range(self._type_num):
    #         self._assign_ops += [tf.assign(self._node_embedding_eval[node_type], self.placeholders['node_embedding_eval_ph'][node_type])]
    #         self._assign_ops += [tf.assign(self._node_cellstates_eval[node_type], self.placeholders['node_cellstates_eval_ph'][node_type])]
    #     self._assign_ops = tf.group(*self._assign_ops)

    # def compute_final_node_representations_peratch_eval(self):
    #     self._create_placeholders_eval()
    #     self._create_variables_eval()
    #     node_vec = self._node_embedding_eval
    #     node_cel = self._node_cellstates_eval
    #     # node_vec_assign_op = [[] for _ in range(self._type_num)]
    #     # node_cel_assign_op = [[] for _ in range(self._type_num)]
    #
    #         # with tf.control_dependencies([node_vec_assign_op[node_type]]):
    #         #     node_vec[node_type] = tf.identity(node_vec[node_type])
    #         # with tf.control_dependencies([node_cel_assign_op[node_type]]):
    #         #     node_cel[node_type] = tf.identity(node_cel[node_type])
    #     # node_vec = self._node_embedding_eval
    #     # node_cel = self._node_cellstates_eval
    #     event_states = tf.zeros([self._h_dim], dtype=tf.float64)
    #     for node_type in range(self._type_num):
    #         states = tf.nn.embedding_lookup(node_vec[node_type],
    #                                         self.placeholders['events_nodes_type_ph'][node_type])
    #         states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
    #         event_states += tf.matmul(states_aggregated, self.weights['edge_weights'][node_type])
    #
    #     for node_type in range(self._type_num):
    #         node_vec_part = tf.dynamic_partition(node_vec[node_type],
    #                                              self.placeholders['event_partition_idx_eval_ph'][node_type],2)
    #         node_cel_part = tf.dynamic_partition(node_cel[node_type],
    #                                              self.placeholders['event_partition_idx_eval_ph'][node_type], 2)
    #         target_embedding = tf.nn.embedding_lookup(node_vec[node_type],
    #                                                   self.placeholders['events_nodes_type_eval_ph'][node_type])
    #         target_celstates = tf.nn.embedding_lookup(node_cel[node_type],
    #                                                   self.placeholders['events_nodes_type_eval_ph'][node_type])
    #
    #         sent_states = tf.tile(event_states, [tf.shape(target_embedding)[0], 1])
    #         if self.params['graph_rnn_cell'].lower() == 'lstm':
    #             _, new_target_states_ch = self.weights['rnn_cells'](inputs=sent_states,
    #                                                                 state=(target_celstates, target_embedding))
    #             new_target_celstates = new_target_states_ch[0]
    #             new_target_embedding = new_target_states_ch[1]
    #
    #
    #         node_vec[node_type] = tf.dynamic_stitch(self.placeholders['event_stitch_idx_eval_ph'][node_type],
    #                                                 [node_vec_part[0], new_target_embedding])
    #         node_cel[node_type] = tf.dynamic_stitch(self.placeholders['event_stitch_idx_eval_ph'][node_type],
    #                                                 [node_cel_part[0], new_target_celstates])
    #         if self.params['use_event_bias']:
    #             event_scores = tf.matmul(event_states, self.weights['event_weights']) + self.weights['event_biases']
    #         else:
    #             event_scores = tf.matmul(event_states, self.weights['event_weights'])
    #         predict = tf.nn.softmax(event_scores)
    #     return predict, node_vec, node_cel

    def test(self):
        test_batches_num = self.test_data.get_batch_num()
        epoch_flag = False
        val_preds = []
        labels = []
        self.node_embedding_cur = [None for node_type in range(self.params['node_type_numbers'])]
        if self.params['graph_rnn_cell'].lower() == 'lstm':
            self.node_cellstates_cur = [None for node_type in range(self.params['node_type_numbers'])]
        for node_type in range(self.params['node_type_numbers']):
            self.node_embedding_cur[node_type] = self._node_embedding[node_type].eval(session=self.sess)
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self.node_cellstates_cur[node_type] = self._node_cellstates[node_type].eval(session=self.sess)
        while not epoch_flag:
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                fetches = [self.ops['embedding'], self.ops['celstates'], self.ops['predict']]
            else:
                fetches = [self.ops['embedding'], self.ops['predict']]
            # batch_feed_dict, epoch_flag = self.test_data.next_batch()
            batch_feed_dict, batch_label, epoch_flag = self.get_batch_feed_dict_test()
            self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                self.node_embedding_cur, self.node_cellstates_cur, predict = self.sess.run(fetches, feed_dict=batch_feed_dict)
            else:
                self.node_embedding_cur, predict = self.sess.run(fetches, feed_dict=batch_feed_dict)
            # val_preds.append(np.argmax(predict, axis=1))
            val_preds.append(predict)
            labels.append(batch_label)
        # precision = metrics.precision_score(labels, val_preds, average=None)
        # recall = metrics.recall_score(labels, val_preds, average=None)
        # f1 = metrics.f1_score(labels, val_preds, average=None)
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        # print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)
        print('mae:%f, rmse:%f' % (mae, rmse))

    # def test(self):
    #     self.test_data = BatchData(self.params, self.test_data)
    #     test_batches_num = self.test_data.get_batch_num()
    #
    #     test_batches_num, test_batches, event_labels = get_eventbatch_seq_eval(self.params, self.test_data)
    #     pred = np.zeros(test_batches_num)
    #     for k in range(test_batches_num):
    #         batch_data = test_batches[k]
    #         fetches = [self.ops['predict'], self.ops['embedding_eval'], self.ops['cellstates_eval']]
    #         batch_feed_dict = self.get_batch_feed_dict_test(batch_data, node_embedding, node_cellstates)
    #         self.sess.run(self._assign_ops, feed_dict=batch_feed_dict)
    #         predict, node_embedding, node_cellstates = self.sess.run(fetches, feed_dict=batch_feed_dict)
    #         top1index = np.argmax(predict)
    #         pred[k] = top1index
    #     precision = metrics.precision_score(event_labels, pred, average=None)
    #     recall = metrics.recall_score(event_labels, pred, average=None)
    #     f1 = metrics.f1_score(event_labels, pred, average=None)
    #     print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)

    def get_batch_feed_dict(self, state):
        batch_feed_dict = {}
        if state == 'train':
            batch_data, batch_label, epoch_flag = self.train_data.next_batch()
        elif state == 'valid':
            batch_data, batch_label, epoch_flag = self.valid_data.next_batch()
            # for node_type in range(self.params['node_type_numbers']):
            #     batch_feed_dict[self.placeholders['node_embedding_eval_ph'][node_type]] = self.node_embedding_cur[node_type]
            #     if self.params['graph_rnn_cell'].lower() == 'lstm':
            #         batch_feed_dict[self.placeholders['node_cellstates_eval_ph'][node_type]] = self.node_cellstates_cur[node_type]
        else:
            print('state wrong')
        for event in range(self.params['batch_event_numbers']):
            for node_type in range(self.params['node_type_numbers']):
                event_partition = np.zeros(self.params['n_nodes_pertype'][node_type])
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(batch_data[event][node_type], dtype=np.int32)
                event_partition[batch_data[event][node_type]] = 1
                batch_feed_dict[self.placeholders['event_partition_idx_ph'][event][node_type]] = event_partition
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][0]] = np.where(event_partition==0)[0].tolist()
                batch_feed_dict[self.placeholders['event_stitch_idx_ph'][event][node_type][1]] = np.where(event_partition==1)[0].tolist()
                batch_feed_dict[self.placeholders['event_labels']] = np.asarray(batch_label, dtype=np.int32)
        return batch_feed_dict, epoch_flag

    def get_batch_feed_dict_test(self):
        batch_feed_dict = {}
        batch_data, batch_label, epoch_flag = self.test_data.next_batch()
        for node_type in range(self.params['node_type_numbers']):
            batch_feed_dict[self.placeholders['node_embedding_eval_ph'][node_type]] = self.node_embedding_cur[node_type]
            if self.params['graph_rnn_cell'].lower() == 'lstm':
                batch_feed_dict[self.placeholders['node_cellstates_eval_ph'][node_type]] = self.node_cellstates_cur[node_type]

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
                batch_feed_dict[self.placeholders['event_labels']] = np.asarray(batch_label, dtype=np.int32)
        return batch_feed_dict, batch_label, epoch_flag

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







