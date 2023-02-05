import random

# import tensorflow.contrib.eager as tfe
import tensorflow as tf

from layers.aggregators import MeanAggregator, AttentionAggregator
from models.basic_model import BasicModel
from utils.inits import glorot_init, zeros_init


# tfe.enable_eager_execution()


class GnnEventModel(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    def _create_placeholders(self):
            self.placeholders['events_nodes_type_ph'] = [[tf.placeholder(tf.int32, [None], name='node_type%i_event%i_ph'%(node_type, event))
                                        for node_type in range(self._type_num)]
                                       for event in range(self._eventnum_batch)]

            self.placeholders['events_negnodes_type_ph'] = [[[tf.placeholder(tf.int32, [None], name='negnode_radio%i_type%i_event%i_ph'%(neg_radio, node_type, event))
                                            for neg_radio in range(self._negative_ratio)]
                                           for node_type in range(self._type_num)]
                                          for event in range(self._eventnum_batch)]

            self.placeholders['node_embedding_ph'] = [tf.placeholder(tf.float64, shape=[self._num_node_type[node_type], self._h_dim], name='node_type%i_embedding_initial_ph'%(node_type))
                                    for node_type in range(self._type_num)]

            self.placeholders['node_cellstates_ph'] = [tf.placeholder(tf.float64, shape=[self._num_node_type[node_type], self._h_dim], name='node_type%i_cellstates_initial_ph'%(node_type))
                                    for node_type in range(self._type_num)]

            self.placeholders['dropout_rate_ph'] = tf.placeholder_with_default(0.0, shape=[])

            self.placeholders['batch_partition_idx_ph'] = [tf.placeholder(tf.int32, shape=[self._num_node_type[node_type]], name='batch_partition_idx_type%i_ph'%(node_type))
                                                            for node_type in range(self._type_num)]

            self.placeholders['event_partition_idx_ph'] = [[tf.placeholder(tf.int32, shape=[None], name='event_partition_idx_type%i_event%i_ph'%(node_type, event))
                                    for node_type in range(self._type_num)]
                                    for event in range(self._eventnum_batch)]

            self.placeholders['batch_stitch_idx_ph'] = [[[
                tf.placeholder(tf.int32, shape=[None]),
                tf.placeholder(tf.int32, shape=[None])
            ] for node_type in range(self._type_num)] for event in range(self._eventnum_batch)]

            self.placeholders['event_stitch_idx_ph'] = [[[
                tf.placeholder(tf.int32, shape=[None]),
                tf.placeholder(tf.int32, shape=[None])
            ] for node_type in range(self._type_num)] for event in range(self._eventnum_batch)]

    def _create_variables(self):

        # self._partition_idx = [tf.get_variable('partition_idx%i'%(node_type), shape=[self._eventnum_batch, self._num_node_type[node_type]], dtype=tf.int32, trainable=False)
        #                        for node_type in range(self._type_num)]

        # self._stitch_idx = []
        cur_seed = random.getrandbits(32)
        self._node_embedding = [tf.get_variable('node_type%i_embedding'%(node_type), shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]
        self._node_cellstates = [tf.get_variable('node_type%i_cellstates'%(node_type), shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]


    # def _get_assign_ops(self):
    #
    #     # self._partition_idx = [tf.assign(self._partition_idx[node_type], self._partition_idx_ph[node_type])
    #     #                        for node_type in range(self._type_num)]
    #     self._node_embedding_assign_op = [[] for _ in range(self._type_num)]
    #     self._node_cellstates_assign_op = [[] for _ in range(self._type_num)]
    #     for node_type in range(self._type_num):
    #         self._node_embedding_assign_op[node_type] = tf.assign(self._node_embedding[0][node_type], self.placeholders['node_embedding_ph'][node_type])
    #
    #         self._node_cellstates_assign_op[node_type] = tf.assign(self._node_cellstates[0][node_type], self.placeholders['node_cellstates_ph'][node_type])
    #
    #         with tf.control_dependencies([self._node_embedding_assign_op[node_type]]):
    #             self._node_embedding[0][node_type] = tf.identity(self._node_embedding[0][node_type])
    #             # self._node_embedding[node_type] = tf.stop_gradient(self._node_embedding[node_type])
    #             # self._node_embedding[node_type] = self._node_embedding[node_type].read_value()
    #             # self._node_embedding[node_type] = tf.stop_gradient(self._node_embedding[node_type])
    #         with tf.control_dependencies([self._node_cellstates_assign_op[node_type]]):
    #             # self._node_cellstates[node_type] = self._node_cellstates[node_type].read_value()
    #             # self._node_cellstates[node_type] = tf.stop_gradient(self._node_cellstates[node_type])
    #             self._node_cellstates[0][node_type] = tf.identity(self._node_cellstates[0][node_type])
    #             # self._node_cellstates[node_type] = tf.stop_gradient(self._node_cellstates[node_type])

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._negative_ratio = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']

        self._create_placeholders()
        self._create_variables()
        # self._get_assign_ops()

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['edge_weights'] = [tf.Variable(glorot_init([self._h_dim, self._h_dim], name='gnn_edge_weights_typ%i' % node_type))
                                        for node_type in range(self._type_num)]

        if self.params['use_type_bias']:
            self.weights['edge_biases'] = [tf.Variable(zeros_init([self._h_dim], name='gnn_edge_biases_typ%i' % node_type))
                for node_type in range(self._type_num)]

        aggregator_type = self.params['aggregator_type'].lower()
        if aggregator_type == 'mean':
            aggregator = MeanAggregator
        elif aggregator_type == 'attention':
            aggregator = AttentionAggregator
        else:
            raise Exception('Unknown aggregator: ', aggregator_type)

        self.weights['aggregators'] = aggregator()

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

    def compute_final_node_representations_perbatch(self):
        loss_pertype = [[None for node_type in range(self._type_num)] for event_id in range(self._eventnum_batch)]
        # node_vec = [[None]*self._type_num]*(self._eventnum_batch+1)
        # node_cel = [[None]*self._type_num]*(self._eventnum_batch+1)
        # node_vec[0] = self._node_embedding
        # node_cel[0] = self._node_cellstates
        # nodes_states = [[]]*(self._eventnum_batch+1)
        # sent_messages = [[]]*(self._eventnum_batch+1)
        for event_id in range(self._eventnum_batch):
            nodes_states = []
            nodes_cellstates = []
            sent_messages = []
            with tf.variable_scope('event_id%i'%(event_id)):
                for node_type in range(self._type_num):
                    nodes_states_event_type = tf.nn.embedding_lookup(self._node_embedding[node_type], self.placeholders['events_nodes_type_ph'][event_id][node_type])
                    nodes_cellstates_event_type = tf.nn.embedding_lookup(self._node_cellstates[node_type], self.placeholders['events_nodes_type_ph'][event_id][node_type])
                    nodes_states_event_type_avg = tf.reduce_mean(nodes_states_event_type, axis=0, keepdims=True)
                    # nodes_cellstates_event_type_avg = tf.reduce_mean(nodes_cellstates_event_type, axis=0, keepdims=True)
                    nodes_states.append(nodes_states_event_type_avg)
                    # nodes_cellstates.append(nodes_cellstates_event_type_avg)
                    message = tf.reduce_mean(tf.matmul(nodes_states_event_type, self.weights['edge_weights'][node_type]), axis=0, keepdims=True)
                    if self.params['use_type_bias']:
                        message += self.weights['edge_biases'][node_type]
                    sent_messages.append(message)
                for node_type in range(self._type_num):
                    # event_partition_idx = tf.squeeze(tf.gather(self.placeholders['partition_idx_ph'][node_type], tf.convert_to_tensor([event_id])))
                    node_vec_part = tf.dynamic_partition(self._node_embedding[node_type], self.placeholders['partition_idx_ph'][event_id][node_type], 2)
                    target_states_event_type = tf.nn.embedding_lookup(self._node_embedding[node_type], self.placeholders['events_nodes_type_ph'][event_id][node_type])
                    target_cellstates_event_type = tf.nn.embedding_lookup(self._node_cellstates[node_type], self.placeholders['events_nodes_type_ph'][event_id][node_type])
                    sent_states_event_type = tf.nn.embedding_lookup(sent_messages, tf.convert_to_tensor([other_type for other_type in range(self._type_num) if other_type != node_type], dtype=tf.int32))
                    sent_states_event_type_aggregated = self.weights['aggregators']((target_states_event_type, sent_states_event_type))

                    if self.params['graph_rnn_cell'].lower() == 'lstm':
                        _, new_target_states_event_type_ch = self.weights['rnn_cells'](inputs=sent_states_event_type_aggregated, state=(target_cellstates_event_type, target_states_event_type))
                        new_target_states_event_type = new_target_states_event_type_ch[1]
                        new_target_cellstates_event_type = new_target_states_event_type_ch[0]

                    target_types_num = tf.shape(self.placeholders['events_nodes_type_ph'][event_id][node_type])[0]
                    context_states = tf.reduce_mean(tf.nn.embedding_lookup(nodes_states, tf.convert_to_tensor([other_type for other_type in range(self._type_num) if other_type != node_type], dtype=tf.int32)), axis=0, keepdims=True)
                    context_states = tf.tile(context_states, [target_types_num, 1])
                    loss_pertype[event_id][node_type] = -tf.reduce_mean(tf.log_sigmoid(tf.reduce_sum(tf.multiply(target_states_event_type, context_states), axis=1)))

                    for neg_radio in range(self._negative_ratio):
                        neg_target_states_event_type = tf.nn.embedding_lookup(self._node_embedding[node_type], self.placeholders['events_negnodes_type_ph'][event_id][node_type])
                        loss_pertype[event_id][node_type] += -tf.reduce_mean(tf.log_sigmoid(-1*tf.reduce_sum(tf.multiply(neg_target_states_event_type, context_states), axis=1)))

                    self._node_embedding[node_type] = tf.dynamic_stitch(self.placeholders['stitch_idx_ph'][event_id][node_type], [node_vec_part[0], new_target_states_event_type])
                    self._node_cellstates[node_type] = tf.dynamic_stitch(self.placeholders['stitch_idx_ph'][event_id][node_type], [node_vec_part[0], new_target_cellstates_event_type])

        loss = tf.reduce_mean(loss_pertype)
        gradient = tf.gradients(loss, target_states_event_type)

        return self._node_embedding, self._node_cellstates, loss, gradient







