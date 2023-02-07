# import tensorflow.contrib.eager as tfe
import tensorflow as tf

from layers.aggregators import MeanAggregator, AttentionAggregator
from models.basic_model import BasicModel
from utils.inits import glorot_init, zeros_init


# tfe.enable_eager_execution()


class GnnModel(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    # @classmethod
    # def default_params(cls):
    #     params = dict(super().default_params())
    #     params.update({
    #         'node_type_numbers': 3,
    #         'batch_event_numbers': 200,
    #         'use_propagation_attention': False,
    #         'use_type_bias': True,
    #         'graph_rnn_cell': 'LSTM',
    #         'graph_rnn_activation': 'tanh',
    #         'graph_state_keep_prob': 0.8,
    #         'negative_ratio': 5,
    #         'table_size': 1e8,
    #     })
    #     return params

    def prepare_specific_graph_model(self) -> None:
        self.h_dim = self.params['hidden_size']
        self.num_node_type = self.params['node_type_numbers']
        self.eventnum_per_batch = self.params['batch_event_numbers']
        self.negative_ratio = self.params['negative_ratio']

        # self.placeholders['cur_node_states'] = tf.placeholder(tf.float32, [None, self.h_dim], name='node_features')
        self.cur_node_states_pertype = [tf.get_variable('embedding_type%i' % (type), [self.params['n_nodes_pertype'][type], self.h_dim]) for type in range(self.num_node_type)]

        self.placeholders['events_nodes'] = [[tf.placeholder(tf.int32, [None], name='nodetype%i_event%i' % (node_type, event))
                                              for node_type in range(self.num_node_type)]
                                              for event in range(self.eventnum_per_batch)]

        self.placeholders['neg_events_nodes'] = [[[tf.placeholder(tf.int32, [None], name='neg_ratio%i_nodetype%i_event%i' % (neg_radio, node_type, event))
                                              for neg_radio in range(self.negative_ratio)]
                                              for node_type in range(self.num_node_type)]
                                              for event in range(self.eventnum_per_batch)]


        # self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['edge_weights'] = [tf.Variable(glorot_init([self.h_dim, self.h_dim], name='gnn_edge_weights_typ%i' % e_typ))
                                        for e_typ in range(self.num_node_type)]

        if self.params['use_type_bias']:
            self.weights['edge_biases'] = [tf.Variable(zeros_init([self.h_dim], name='gnn_edge_biases_typ%i' % e_typ))
                for e_typ in range(self.num_node_type)]
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
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim, activation=activation_fun)
        elif cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(self.h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(self.h_dim, activation=activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.params['graph_state_keep_prob'])

        self.weights['rnn_cells'] = cell

    def compute_final_node_representations_perbatch(self):

        # cur_node_states = self.placeholders['cur_node_states']

        # event_node_ta = tf.TensorArray(tf.int32, infer_shape=False,
        #                                element_shape=[None],
        #                                size=self.eventnum_per_batch*self.num_node_type,
        #                                clear_after_read=False)
        #
        # neg_event_node_ta = tf.TensorArray(tf.int32, infer_shape=False,
        #                                    element_shape=[None],
        #                                    size=self.eventnum_per_batch*self.num_node_type*self.negative_ratio,
        #                                    clear_after_read=False)

        # node_states_ta_pertype = [tf.TensorArray(tf.float32, infer_shape=False,
        #                                              element_shape=[self.h_dim],
        #                                              size=tf.shape(self.cur_node_states_pertype[type])[0],
        #                                              clear_after_read=True) for type in range(self.num_node_type)]
        #
        # new_node_states_ta_pertype = [tf.TensorArray(tf.float32, infer_shape=False,
        #                                     element_shape=[self.h_dim],
        #                                     size = tf.shape(self.cur_node_states_pertype[type])[0],
        #                                     clear_after_read=False) for type in range(self.num_node_type)]


        # new_node_states_ta = new_node_states_ta.scatter(self.placeholders['initial_nodes'][batch_id],
        #                                                 tf.gather(cur_node_states, self.placeholders['initial_nodes'][batch_id]))

        new_node_states_ta_pertype = [new_node_states_ta_pertype[node_type].unstack(self.cur_node_states_pertype[node_type]) for node_type in range(self.num_node_type)]
        # new_node_states_ta_pertype = [new_node_states_ta_pertype[node_type].scatter(self.cur_node_states_pertype[node_type]) for node_type in range(self.num_node_type)]
        loss_pertype = [0.0 for node_type in range(self.num_node_type)]
        for event_id in range(self.eventnum_per_batch):
            for node_type in range(self.num_node_type):
                event_node_ta = event_node_ta.write(event_id*self.num_node_type+node_type, self.placeholders['events_nodes'][event_id][node_type])
                for neg_radio in range(self.negative_ratio):
                    neg_event_node_ta = neg_event_node_ta.write(event_id*self.num_node_type*self.negative_ratio + node_type*self.negative_ratio + neg_radio, self.placeholders['neg_events_nodes'][event_id][node_type][neg_radio])

        def eventEnvolve(event_id, new_node_states_ta_pertype, loss_pertype):
            # sent_messages = []
            sent_messages_ta = tf.TensorArray(tf.float32, infer_shape=False, element_shape=[None], size=self.num_node_type, clear_after_read=False)
            # types_states = []
            states_pertypes_ta = tf.TensorArray(tf.float32, infer_shape=False, element_shape=[None], size=self.num_node_type, clear_after_read=False)
            for node_type in range(self.num_node_type):
                nodes_states_pertypes = new_node_states_ta_pertype[node_type].gather(event_node_ta.read(event_id*self.num_node_type+node_type))
                average_state_pertype = tf.reduce_mean(nodes_states_pertypes, 0)
                # average_state_pertype = tf.expand_dims(tf.reduce_mean(nodes_states_pertypes, 0), 1)
                # types_states.append(average_state_pertype)
                states_pertypes_ta.write(node_type, average_state_pertype)
                message = tf.reduce_mean(tf.matmul(nodes_states_pertypes, self.weights['edge_weights'][node_type]), 0)
                if self.params['use_type_bias']:
                    message += self.weights['edge_biases'][node_type]
                # message = tf.expand_dims(tf.reduce_mean(messages, 0), 1)
                # sent_messages.append(message)
                sent_messages_ta.write(node_type, tf.nn.relu(message))
            # sent_messages = tf.concat(sent_messages, axis=1)
            # types_states = tf.concat(types_states, axis=1)

            # loss_pertype = [0 for type in range(self.num_node_type)]
            for node_type in range(self.num_node_type):
                event_node_type = event_node_ta.read(event_id*self.num_node_type+node_type)
                target_state_pertype = new_node_states_ta_pertype[node_type].gather(event_node_type)
                received_message_pertypes = sent_messages_ta.gather(tf.convert_to_tensor([other_type for other_type in range(self.num_node_type) if other_type != node_type], dtype=tf.int32))
                received_message_aggregated = self.weights['aggregators']((target_state_pertype, received_message_pertypes))

                target_state_pertype.set_shape([None, self.h_dim])
                received_message_aggregated.set_shape([None, self.h_dim])

                if self.params['graph_rnn_cell'].lower() == 'lstm':
                    _, new_target_state_pertype_ch = self.weights['rnn_cells'](inputs=received_message_aggregated, state=(target_state_pertype, target_state_pertype))
                    new_target_state_pertype = new_target_state_pertype_ch[1]

                # tf.nn.sampled_softmax_loss
                # loss_pertype = tf.map_fn(fn=lambda target: calculate_loss(node_type, target), elems=event_node_type, dtype=tf.float32)
                target_vecs_num = tf.shape(event_node_type)[0]
                context_states = tf.reduce_mean(states_pertypes_ta.gather(tf.convert_to_tensor([other_type for other_type in range(self.num_node_type) if other_type != node_type], dtype=tf.int32)), axis=0, keepdims=True)
                context_states = tf.tile(context_states, [target_vecs_num,1])
                loss_pertype[node_type] += -tf.reduce_mean(tf.log_sigmoid(tf.reduce_sum(tf.multiply(target_state_pertype, context_states), axis=1)))
                for neg_radio in range(self.negative_ratio):
                    neg_target_state_pertype = new_node_states_ta_pertype[node_type].gather(neg_event_node_ta.read(event_id*self.num_node_type*self.negative_ratio+node_type*self.negative_ratio+neg_radio))
                    loss_pertype[node_type] += -tf.reduce_mean(tf.log_sigmoid(-1*tf.reduce_sum(tf.multiply(neg_target_state_pertype, context_states), axis=1)))

                # new_node_states_ta_pertype[node_type] = new_node_states_ta_pertype[node_type].scatter(event_node_type, new_target_state_pertype)
                # loss = np.mean(loss_pertype)
            return (event_id+1, new_node_states_ta_pertype, loss_pertype)

            # def calculate_loss(node_type, target):
            #     target_state = tf.gather(new_node_states_ta_pertype[node_type], target)
            #     context_state = tf.gather(types_states_ta, [other_type for other_type in range(self.num_node_type) if other_type != node_type])
            #     average_context_states = tf.reduce_mean(context_states)
            #     losses.append(tf.log_sigmoid(tf.multiply(target_state, average_context_states)))
            #     negative_ratio = self.params['negative_ratio']
            #     table_size = self.params['table_size']
            #     for i in range(negative_ratio+1):
            #         negative_state = new_node_states_ta_pertype[node_type].gather(self.sampling_table[type][random.randint(0, table_size-1)])
            #         losses.append(tf.log_sigmoid(-1*tf.multiply(negative_state, average_context_states)))

        def is_done(event_id, new_node_states_ta_unused, losses_unused):
            return event_id < self.params['batch_event_numbers']

        _, new_node_states_ta_pertype, loss_pertype = tf.while_loop(cond=is_done,
                                              body=eventEnvolve,
                                              loop_vars=[tf.constant(0), new_node_states_ta_pertype, loss_pertype], parallel_iterations=1)

        cur_node_states_pertype = [new_node_states_ta_pertype[node_type].stack() for node_type in range(self.num_node_type)]
        loss = tf.reduce_mean(loss_pertype)

        return cur_node_states_pertype, loss

    # def gen_sampling_table_pertype(self, batch_data):
    #     table_size = self.params['table_size']
    #     power = self.params['neg_power']
    #     # power = 0.75
    #     for type in range(self.num_node_type):
    #         numNodesPerType = len(self.hegraph.nodes[type])
    #         node_hyperdegree = np.zeros(numNodesPerType)
    #         for node in self.hegraph.nodes[type]:
    #             node_hyperdegree[node] = self.hegraph.nodes_degree[type][node]
    #         norm = sum([math.pow(node_hyperdegree[i], power)] for i in range(numNodesPerType))
    #         self.sampling_table[type] = np.zeros(int(table_size), dtype=np.uint32)
    #         p = 0
    #         i = 0
    #         for j in range(numNodesPerType):
    #             p += float(math.pow(node_hyperdegree[j], power)) / norm
    #             while i < table_size and float(i) / table_size < p:
    #                 self.sampling_table[type][i] = j
    #                 i += 1

    # def gen_sampling_table_pertype(self, batch_data):





