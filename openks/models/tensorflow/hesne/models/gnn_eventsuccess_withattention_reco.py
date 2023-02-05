import random
import sys
import time
import random

import sklearn.metrics as metrics
from layers.layers import *
from layers.aggregators import *
from models.basic_model import BasicModel
from utils.data_manager import *
from scipy import stats


# from scipy.sparse import linalg

class GnnEventModel_withattention_reco(BasicModel):
    def neg_type__init__(self, args):
        super().__init__(args)

    def get_log_file(self):
        log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                   + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                   + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + '_' + str(self.params['his_type']) + '_' \
                   + str(self.params['max_his_num']) + '_n_' + str(self.params['negative_ratio']) + '_ns_' + str(self.params['negative_sampling']) + '_witht_reco.log'
        return log_file

    def get_checkpoint_dir(self):
        checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                         + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                         + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + '_' + str(self.params['his_type']) + '_' \
                         + str(self.params['max_his_num']) + '_n_' + str(self.params['negative_ratio']) + '_ns_' + str(self.params['negative_sampling']) + '_witht_reco'
        return checkpoint_dir

    def make_model(self):
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['node_vec'], self.ops['node_cel'] = self.build_specific_graph_model()
            self.ops['node_vec_eval'], self.ops['node_cel_eval'], self.ops['predict'] = self.build_specific_evo_graph_model()
            self.ops['neg_predict'] = self.build_specific_eval_graph_model()
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

            self.placeholders['events_nodes_history_deltatime_ph'] = [tf.placeholder(tf.float64, shape=[None, self._max_his_num],
                                                        name='event%i_histrory_deltatime_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['events_partition_idx_ph'] = [tf.placeholder(tf.int32, shape=[self._num_node],
                                                        name='event%i_partition_idx_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['nodes_embedding_ph'] = tf.placeholder(tf.float64,
                                                    shape=[self._num_node, self._h_dim], name='node_embedding_ph')

            self.placeholders['nodes_cellstates_ph'] = tf.placeholder(tf.float64,
                                                    shape=[self._num_node, self._h_dim], name='node_cellstates_ph')

            self.placeholders['is_train'] = tf.placeholder(tf.bool, name='is_train')

            self.placeholders['is_init'] = tf.placeholder(tf.bool, name='is_init')

            self.placeholders['keep_prob'] = tf.placeholder(tf.float64, name='keep_prob')

            self.placeholders['has_neighbor'] = [tf.placeholder(tf.bool, shape=[None],
                                            name='event%i_hasneighbor_ph'%event)
                                            for event in range(self._eventnum_batch)]

            self.placeholders['events_deltatime_ph'] = [tf.placeholder(tf.float64,
                                                    shape=[None], name='event%i_deltatime_ph'%event)
                                                    for event in range(self._eventnum_batch)]

###########test placeholder###########################
            self.placeholders['event_nodes_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='event_eval_ph')

            self.placeholders['event_nodes_type_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='event_type_eval_ph')

            self.placeholders['event_nodes_history_eval_ph'] = tf.placeholder(tf.int32,
                                                        shape=[None, self._max_his_num, self._type_num], name='event_history_eval_ph')

            self.placeholders['event_nodes_history_deltatime_eval_ph'] = tf.placeholder(tf.float64,
                                                        shape=[None, self._max_his_num], name='event_history_deltatime_eval_ph')

            self.placeholders['event_partition_idx_eval_ph'] = tf.placeholder(tf.int32, shape=[self._num_node],
                                                            name='event_partition_idx_eval_ph')

            self.placeholders['negevent_nodes_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='negevent_eval_ph')

            self.placeholders['negevent_nodes_type_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='negevent_type_eval_ph')

            self.placeholders['negevent_nodes_history_eval_ph'] = tf.placeholder(tf.int32,
                                                shape=[None, self._max_his_num, self._type_num], name='negevent_history_eval_ph')

            self.placeholders['negevent_nodes_history_deltatime_eval_ph'] = tf.placeholder(tf.float64,
                                                shape=[None, self._max_his_num], name='negevent_history_deltatime_eval_ph')

            self.placeholders['has_neighbor_eval'] = tf.placeholder(tf.bool, shape=[None], name='hasneighbor_eval_ph')

            self.placeholders['has_neighbor_neg_eval'] = tf.placeholder(tf.bool, shape=[None], name='hasneighbor_neg_eval_ph')

            self.placeholders['event_deltatime_eval_ph'] = tf.placeholder(tf.float64, shape=[None], name='event_deltatime_eval_ph')
######################################################

    def _create_variables(self):
        cur_seed = random.getrandbits(32)

        self._embedding_init = tf.get_variable('nodes_embedding_init', shape=[1, self._h_dim],
                                                dtype=tf.float64, trainable=True,
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

        self._cellstates_init = tf.get_variable('nodes_cellstates_init', shape=[1, self._h_dim],
                                                 dtype=tf.float64, trainable=True,
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                                 # initializer=tf.zeros_initializer())

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
        self.weights['aggregator'] = aggregator(self._h_dim, keep=self.placeholders['keep_prob'], weight_decay=self._weight_decay)
        self.weights['type_weights_scalar'] = tf.Variable(tf.ones([self._type_num], dtype=tf.float64), trainable=True)
        self.weights['rnn_cells'] = {}
        if self.params['use_different_cell']:
            for node_type in range(self._type_num):
                cell = tf.nn.rnn_cell.LSTMCell(self._h_dim, activation=self.activation_fun, use_peepholes=True)
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['keep_prob'])
                if self._keep < 1.0:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.placeholders['keep_prob'])
                self.weights['rnn_cells'][node_type] = cell
        else:
            cell = tf.nn.rnn_cell.LSTMCell(self._h_dim, activation=self.activation_fun, use_peepholes=True)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['keep_prob'])
            if self._keep < 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.placeholders['keep_prob'])
            for node_type in range(self._type_num):
                self.weights['rnn_cells'][node_type] = cell

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._neg_num = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']
        self._num_node = sum(self._num_node_type)
        self._sub_events_train, self._events_time_train = self.train_data.get_subevents()
        self._sub_events_valid, self._events_time_valid = self.valid_data.get_subevents()
        self._sub_events_test, self._events_time_test = self.test_data.get_subevents()
        self._max_his_num = self.params['max_his_num']
        self._his_type = self.params['his_type']
        self._keep = self.params['keep_prob']
        self._weight_decay = self.params['weight_decay']
        # self._istraining = self.params['is_training']
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
        self.triangularize_layer = Triangularize()
        self.attention_concat_layer = tf.layers.Dense(units=self._h_dim, use_bias=False, \
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay))
        for event_id in range(self._eventnum_batch):
            node_vec_part = tf.dynamic_partition(node_vec[event_id],
                                                 self.placeholders['events_partition_idx_ph'][event_id],
                                                 self._type_num + 1)
            node_cel_part = tf.dynamic_partition(node_cel[event_id],
                                                 self.placeholders['events_partition_idx_ph'][event_id],
                                                 self._type_num + 1)
            dy_states = tf.nn.embedding_lookup(node_vec[event_id], self.placeholders['events_nodes_ph'][event_id])
            his_events_states = tf.nn.embedding_lookup(node_vec[event_id], self.placeholders['events_nodes_history_ph'][event_id])
            his_events_deltatime = self.placeholders['events_nodes_history_deltatime_ph'][event_id]
            his_events_states = tf.einsum('nmth,t->nmh', his_events_states, self.weights['type_weights_scalar'])
            his_states = self.weights['aggregator']((dy_states, his_events_states, his_events_deltatime))
            hisconcat_states = tf.concat([dy_states, his_states], 1)
            hisconcat_states = self.attention_concat_layer(hisconcat_states)
            # hisconcat_states = self.attention_concat_layer(tf.concat([dy_states, his_states], 1))
            hisconcat_states = tf.where(self.placeholders['has_neighbor'][event_id], hisconcat_states, dy_states)
            event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['events_nodes_type_ph'][event_id])
            _, _, type_count = tf.unique_with_counts(self.placeholders['events_nodes_type_ph'][event_id])
            type_count = tf.cast(type_count, tf.float64)
            count_weights = tf.nn.embedding_lookup(tf.reciprocal(type_count),
                                                   self.placeholders['events_nodes_type_ph'][event_id])
            event_weights = tf.multiply(event_weights, count_weights)
            event_states = hisconcat_states*tf.expand_dims(event_weights, 1)
            send_states = tf.reduce_sum(event_states, axis=0, keepdims=True)
            new_target_embedding = [node_vec_part[0]]
            new_target_celstates = [node_cel_part[0]]
            node_detatime_part = tf.dynamic_partition(self.placeholders['events_deltatime_ph'][event_id],
                                            self.placeholders['events_nodes_type_ph'][event_id], self._type_num)
            for node_type in range(self._type_num):
                target_embedding = node_vec_part[node_type+1]
                target_celstates = node_cel_part[node_type+1]
                target_deltatime = tf.expand_dims(node_detatime_part[node_type], 1)
                # time_states = self.deltatime_layer(target_deltatime)
                send_states_pertype = tf.tile(send_states, [tf.shape(target_embedding)[0], 1])
                # send_states_pertype = tf.concat([send_states_pertype, time_states], 1)
                _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=send_states_pertype,
                                                                    state=(target_celstates, target_embedding, target_deltatime))
                new_target_celstates.append(new_target_states_ch[0])
                new_target_embedding.append(new_target_states_ch[1])
            condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                    self.placeholders['events_partition_idx_ph'][event_id], self._type_num+1)
            node_cel[event_id+1] = tf.dynamic_stitch(condition_indices, new_target_celstates)
            node_vec[event_id+1] = tf.dynamic_stitch(condition_indices, new_target_embedding)
        return node_vec[event_id+1], node_cel[event_id+1]

    def build_specific_evo_graph_model(self):
        node_vec_part = tf.dynamic_partition(self.placeholders['nodes_embedding_ph'],
                                             self.placeholders['event_partition_idx_eval_ph'], self._type_num + 1)
        node_cel_part = tf.dynamic_partition(self.placeholders['nodes_cellstates_ph'],
                                             self.placeholders['event_partition_idx_eval_ph'], self._type_num + 1)
        dy_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'], self.placeholders['event_nodes_eval_ph'])
        his_events_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'],
                                                   self.placeholders['event_nodes_history_eval_ph'])
        his_events_deltatime = self.placeholders['event_nodes_history_deltatime_eval_ph']
        his_events_states = tf.einsum('nmth,t->nmh', his_events_states, self.weights['type_weights_scalar'])
        his_states = self.weights['aggregator']((dy_states, his_events_states, his_events_deltatime))
        # hisconcat_states = self.attention_concat_layer(tf.concat([dy_states, his_states], 1))
        hisconcat_states = tf.concat([dy_states, his_states], 1)
        hisconcat_states = self.attention_concat_layer(hisconcat_states)
        hisconcat_states = tf.where(self.placeholders['has_neighbor_eval'], hisconcat_states, dy_states)
        event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['event_nodes_type_eval_ph'])
        _, _, type_count = tf.unique_with_counts(self.placeholders['event_nodes_type_eval_ph'])
        type_count = tf.cast(type_count, tf.float64)
        count_weights = tf.nn.embedding_lookup(tf.reciprocal(type_count),
                                               self.placeholders['event_nodes_type_eval_ph'])
        event_weights = tf.multiply(event_weights, count_weights)
        event_states = hisconcat_states * tf.expand_dims(event_weights, 1)
        send_states = tf.reduce_sum(event_states, axis=0, keepdims=True)
        new_target_embedding = [node_vec_part[0]]
        new_target_celstates = [node_cel_part[0]]
        node_detatime_part = tf.dynamic_partition(self.placeholders['event_deltatime_eval_ph'],
                                            self.placeholders['event_nodes_type_eval_ph'], self._type_num)
        for node_type in range(self._type_num):
            target_embedding = node_vec_part[node_type+1]
            target_celstates = node_cel_part[node_type+1]
            target_deltatime = tf.expand_dims(node_detatime_part[node_type], 1)
            # time_states = self.deltatime_layer(target_deltatime)
            send_states_pertype = tf.tile(send_states, [tf.shape(target_embedding)[0], 1])
            # send_states_pertype = tf.concat([send_states_pertype, time_states], 1)
            _, new_target_states_ch = self.weights['rnn_cells'][node_type](inputs=send_states_pertype,
                                                            state=(target_celstates, target_embedding, target_deltatime))
            new_target_celstates.append(new_target_states_ch[0])
            new_target_embedding.append(new_target_states_ch[1])

        condition_indices = tf.dynamic_partition(tf.range(self._num_node),
                                                 self.placeholders['event_partition_idx_eval_ph'],
                                                 self._type_num + 1)

        node_cel = tf.dynamic_stitch(condition_indices, new_target_celstates)
        node_vec = tf.dynamic_stitch(condition_indices, new_target_embedding)
        ###pairwise layer to predict
        event_scores = event_states
        event_scores = tf.expand_dims(event_scores, 0)
        event_scores_h = tf.matmul(event_scores, event_scores, transpose_b=True)
        event_scores_h = self.triangularize_layer(event_scores_h)
        event_scores_h = tf.layers.flatten(event_scores_h)
        y_pred = tf.reduce_sum(event_scores_h, 1, keepdims=True)
        predict = tf.sigmoid(tf.squeeze(y_pred))
        return node_vec, node_cel, predict

    def build_specific_eval_graph_model(self):
        neg_dy_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'], self.placeholders['negevent_nodes_eval_ph'])
        neg_his_events_states = tf.nn.embedding_lookup(self.placeholders['nodes_embedding_ph'],
                                                       self.placeholders['negevent_nodes_history_eval_ph'])
        neg_his_events_deltatime = self.placeholders['negevent_nodes_history_deltatime_eval_ph']
        neg_his_events_states = tf.einsum('nmth,t->nmh', neg_his_events_states, self.weights['type_weights_scalar'])
        neg_his_states = self.weights['aggregator']((neg_dy_states, neg_his_events_states, neg_his_events_deltatime))
        # neg_hisconcat_states = self.attention_concat_layer(tf.concat([neg_dy_states, neg_his_states], 1))
        neg_hisconcat_states = tf.concat([neg_dy_states, neg_his_states], 1)
        neg_hisconcat_states = self.attention_concat_layer(neg_hisconcat_states)
        neg_hisconcat_states = tf.where(self.placeholders['has_neighbor_neg_eval'], neg_hisconcat_states, neg_dy_states)
        neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['negevent_nodes_type_eval_ph'])
        _, _, type_count = tf.unique_with_counts(self.placeholders['negevent_nodes_type_eval_ph'])
        type_count = tf.cast(type_count, tf.float64)
        count_weights = tf.nn.embedding_lookup(tf.reciprocal(type_count),
                                               self.placeholders['negevent_nodes_type_eval_ph'])
        neg_event_weights = tf.multiply(neg_event_weights, count_weights)
        neg_event_states = neg_hisconcat_states * tf.expand_dims(neg_event_weights, 1)
        ###pairwise layer to predict
        neg_event_scores = neg_event_states
        neg_event_scores = tf.expand_dims(neg_event_scores, 0)
        neg_event_scores_h = tf.matmul(neg_event_scores, neg_event_scores, transpose_b=True)
        neg_event_scores_h = self.triangularize_layer(neg_event_scores_h)
        neg_event_scores_h = tf.layers.flatten(neg_event_scores_h)
        neg_y_pred = tf.reduce_sum(neg_event_scores_h, 1, keepdims=True)
        neg_predict = tf.sigmoid(tf.squeeze(neg_y_pred))
        return neg_predict

    def train(self):
        # train_batches_num = self.train_data.get_batch_num()
        # print('batches'+str(train_batches_num))
        for epoch in range(self.params['num_epochs']):
            self.node_embedding_cur = np.zeros([self._num_node, self._h_dim], dtype=np.float64)
            self.node_cellstates_cur = np.zeros([self._num_node, self._h_dim], dtype=np.float64)
            self.node_his_event = {node:[] for node in range(self._num_node)}
            is_init = True
            epoch_flag = False
            print('start epoch %i'%(epoch))
            while not epoch_flag:
                batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train', is_init)
                # fetches = [self.ops['node_vec'], self.ops['node_cel']]
                # self.node_embedding_cur, self.node_cellstates_cur=self.sess.run(fetches, feed_dict=batch_feed_dict)
                is_init = False
            self.node_embedding_cur = np.load(self.params['init_from']+'_embedding.npy')
            self.node_cellstates_cur = np.load(self.params['init_from']+'_cellstates.npy')
            print('start valid')
            self.validation()
            print('start test')
            self.test()


    def validation(self):
        epoch_flag = False
        is_init = False
        valid_batches_num = self.valid_data.get_batch_num()
        while not epoch_flag:
            fetches = [self.ops['node_vec'], self.ops['node_cel']]
            feed_dict_valid, epoch_flag = self.get_batch_feed_dict('valid', is_init)
            self.node_embedding_cur, self.node_cellstates_cur = self.sess.run(fetches, feed_dict=feed_dict_valid)

    def check_rank(self, pred_list, value):
        return len([x for x in pred_list if x>value])+1

    def test(self):
        self.test_data.batch_size = 1
        test_batches_num = self.test_data.get_batch_num()
        print('test nums:'+str(test_batches_num))
        epoch_flag = False
        rank_list = []
        val_preds = []
        labels = []
        # test_num = self.params['test_num']
        # test_ids = random.sample(range(test_batches_num), test_num)
        #mag
        test_ids = range(test_batches_num)
        test_idx = 0
        while not epoch_flag:
            feed_dict_evo, eval_data, epoch_flag = self.get_feed_dict_evo(is_init=False)
            fetches_evo = [self.ops['node_vec_eval'], self.ops['node_cel_eval'], self.ops['predict']]
            if test_idx not in test_ids:
                self.node_embedding_cur, self.node_cellstates_cur, predict = self.sess.run(fetches_evo, feed_dict=feed_dict_evo)
            else:
                neg_predict_score = []
                neg_flag = False
                neg_type = self.params['neg_type']
                start_idx = 0
                replace_node = random.choice(eval_data[neg_type])
                eval_data_neg_list = self.sample_negbatch_events_eval(eval_data, neg_type, replace_node)
                # eval_data_neg_list = self.sample_negbatch_events_eval1([eval_data], 1)
                while not neg_flag:
                    fetches_eval = [self.ops['neg_predict']]
                    feed_dict_eval, start_idx, neg_flag = self.get_feed_dict_eval(eval_data_neg_list, start_idx, is_init=False)
                    neg_predict = self.sess.run(fetches_eval, feed_dict=feed_dict_eval)
                    neg_predict_score.extend(neg_predict)
                self.node_embedding_cur, self.node_cellstates_cur, predict = self.sess.run(fetches_evo, feed_dict=feed_dict_evo)
                val_preds.append(predict)
                val_preds.append(random.choice(neg_predict_score))
                labels.append(1)
                labels.append(0)
                rank_list.append(self.check_rank(neg_predict_score, predict))
                # print(predict)
                # print(neg_predict_score[:100])
                # time.sleep(10)
                # print(rank_list)
            test_idx += 1
            for node_type in range(self._type_num):
                for node in eval_data[node_type]:
                    self.node_his_event[node].append(eval_data[-1])
        print(labels[:100])
        print(val_preds[:100])
        print('hmean%f'%stats.hmean(rank_list))
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        fpr, tpr, thresholds = metrics.roc_curve(labels, val_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc = metrics.roc_auc_score(labels, val_preds)
        ap = metrics.average_precision_score(labels, val_preds)
        embedding_path = self.params['init_from']+'_embedding_test.npy'
        np.save(embedding_path, self.node_embedding_cur)
        print('mae:%f, rmse:%f, auc:%f, ap:%f' % (mae, rmse, auc, ap))

    def sample_subevent_fromhis(self, event_data, node_his_event, sub_events_list, events_time_list, his_type):
        if his_type == 'random':
            return self.sample_subevent_fromhis_random(event_data, node_his_event, sub_events_list, events_time_list)
        elif his_type == 'last':
            return self.sample_subevent_fromhis_last(event_data, node_his_event, sub_events_list, events_time_list)
        else:
            print('history type wrong')

    # sample last subevent
    def sample_subevent_fromhis_last(self, event_data, node_his_event, sub_events_list, events_time_list):
        type_data = []
        for node_type in range(self._type_num):
            type_data.extend(event_data[node_type])
        sampled_his_event = [[] for _ in range(len(type_data))]
        isneighbor = [True for _ in range(len(type_data))]
        event_deltatime = [0.0 for _ in range(len(type_data))]
        his_event_deltatime = [[] for _ in range(len(type_data))]
        for node_index in range(len(type_data)):
            his_len = len(node_his_event[type_data[node_index]])
            if his_len == 0:
                sub_his_event = []
                sub_his_event_deltatime = []
                sub_events = event_data[-2]
                while len(sub_his_event)<self._max_his_num:
                    if len(sub_his_event)+len(sub_events)<self._max_his_num:
                        sub_his_event.extend(sub_events)
                        sub_his_event_deltatime.extend([0]*len(sub_events))
                    elif len(sub_his_event)+len(sub_events)==self._max_his_num:
                        sub_his_event.extend(sub_events)
                        sub_his_event_deltatime.extend([0]*len(sub_events))
                        break
                    else:
                        extra = self._max_his_num - len(sub_his_event)
                        # sub_his_event.extend(sub_events[:-extra])
                        selected_index = random.sample(range(len(sub_events)), extra)
                        sub_his_event.extend([sub_events[i] for i in selected_index])
                        sub_his_event_deltatime.extend([0]*extra)
                        break
                sampled_his_event[node_index] = sub_his_event
                his_event_deltatime[node_index] = sub_his_event_deltatime
                isneighbor[node_index] = False
                event_deltatime[node_index] = 0.0
            else:
                sub_his_event = []
                sub_his_event_deltatime = []
                sub_selected_hisevent = []
                sub_selected_hisdeltatime = []
                event_his = node_his_event[type_data[node_index]]
                event_time = event_data[-3]
                for his_index in range(len(event_his)):
                    sub_events = sub_events_list[event_his[his_index]]
                    his_event_time = events_time_list[event_his[his_index]]
                    sub_his_event.extend(sub_events)
                    sub_his_event_deltatime.extend([event_time-his_event_time]*len(sub_events))
                while len(sub_selected_hisevent)<self._max_his_num:
                    extra = self._max_his_num - len(sub_selected_hisevent)
                    if extra>len(sub_his_event):
                        sub_selected_hisevent.extend(sub_his_event)
                        sub_selected_hisdeltatime.extend(sub_his_event_deltatime)
                    else:
                        # selected_index = random.sample(range(len(sub_his_event)), extra)
                        # sub_selected_hisevent.extend([sub_his_event[i] for i in selected_index])
                        # sub_selected_hisdeltatime.extend([sub_his_event_deltatime[i] for i in selected_index])
                        sub_selected_hisevent.extend(sub_his_event[-extra:])
                        sub_selected_hisdeltatime.extend(sub_his_event_deltatime[-extra:])
                sampled_his_event[node_index] = sub_selected_hisevent
                his_event_deltatime[node_index] = sub_selected_hisdeltatime
                isneighbor[node_index] = True
                event_deltatime[node_index] = event_time-events_time_list[event_his[-1]]
        return sampled_his_event, his_event_deltatime, isneighbor, event_deltatime

    #random sample subevent
    def sample_subevent_fromhis_random(self, event_data, node_his_event, sub_events_list, events_time_list):
        type_data = []
        for node_type in range(self._type_num):
            type_data.extend(event_data[node_type])
        sampled_his_event = [[] for _ in range(len(type_data))]
        isneighbor = [True for _ in range(len(type_data))]
        event_deltatime = [0.0 for _ in range(len(type_data))]
        his_event_deltatime = [[] for _ in range(len(type_data))]
        for node_index in range(len(type_data)):
            his_len = len(node_his_event[type_data[node_index]])
            if his_len == 0:
                sub_his_event = []
                sub_his_event_deltatime = []
                sub_events = event_data[-2]
                while len(sub_his_event)<self._max_his_num:
                    if len(sub_his_event)+len(sub_events)<self._max_his_num:
                        sub_his_event.extend(sub_events)
                        sub_his_event_deltatime.extend([0]*len(sub_events))
                    elif len(sub_his_event)+len(sub_events)==self._max_his_num:
                        sub_his_event.extend(sub_events)
                        sub_his_event_deltatime.extend([0]*len(sub_events))
                        break
                    else:
                        extra = self._max_his_num - len(sub_his_event)
                        selected_index = random.sample(range(len(sub_events)), extra)
                        sub_his_event.extend([sub_events[i] for i in selected_index])
                        sub_his_event_deltatime.extend([0]*extra)
                        break
                sampled_his_event[node_index] = sub_his_event
                his_event_deltatime[node_index] = sub_his_event_deltatime
                isneighbor[node_index] = False
                event_deltatime[node_index] = 0.0
            else:
                sub_his_event = []
                sub_his_event_deltatime = []
                sub_selected_hisevent = []
                sub_selected_hisdeltatime = []
                event_his = node_his_event[type_data[node_index]]
                event_time = event_data[-3]
                for his_index in range(len(event_his)):
                    sub_events = sub_events_list[event_his[his_index]]
                    his_event_time = events_time_list[event_his[his_index]]
                    sub_his_event.extend(sub_events)
                    sub_his_event_deltatime.extend([event_time-his_event_time]*len(sub_events))
                while len(sub_selected_hisevent)<self._max_his_num:
                    extra = self._max_his_num - len(sub_selected_hisevent)
                    if extra>len(sub_his_event):
                        sub_selected_hisevent.extend(sub_his_event)
                        sub_selected_hisdeltatime.extend(sub_his_event_deltatime)
                    else:
                        selected_index = random.sample(range(len(sub_his_event)), extra)
                        sub_selected_hisevent.extend([sub_his_event[i] for i in selected_index])
                        sub_selected_hisdeltatime.extend([sub_his_event_deltatime[i] for i in selected_index])
                sampled_his_event[node_index] = sub_selected_hisevent
                his_event_deltatime[node_index] = sub_selected_hisdeltatime
                isneighbor[node_index] = True
                event_deltatime[node_index] = event_time-events_time_list[event_his[-1]]
        return sampled_his_event, his_event_deltatime, isneighbor, event_deltatime

    def get_batch_feed_dict(self, state, is_init):
        batch_feed_dict = {}
        batch_feed_dict[self.placeholders['is_init']] = is_init
        if state == 'train':
            batch_data, epoch_flag = self.train_data.next_batch()
            sub_events = self._sub_events_train
            events_time = self._events_time_train
            batch_feed_dict[self.placeholders['keep_prob']] = self._keep
        elif state == 'valid':
            batch_data, epoch_flag = self.valid_data.next_batch()
            sub_events = {**self._sub_events_train, **self._sub_events_valid}
            events_time = {**self._events_time_train, **self._events_time_valid}
            batch_feed_dict[self.placeholders['keep_prob']] = self._keep
        else:
            print('state wrong')
        batch_feed_dict[self.placeholders['nodes_embedding_ph']] = self.node_embedding_cur
        batch_feed_dict[self.placeholders['nodes_cellstates_ph']] = self.node_cellstates_cur
        for event in range(self._eventnum_batch):
            ###############record history event for each node###################
            event_partition = np.zeros(self._num_node, dtype=np.int32)
            event_data = []
            event_data_type = []
            for node_type in range(self._type_num):
                event_data.extend(batch_data[event][node_type])
                event_data_type.extend([node_type]*len(batch_data[event][node_type]))
                event_partition[batch_data[event][node_type]] = node_type + 1
                # for node in batch_data[event][node_type]:
                #     self.node_his_event[node].append(batch_data[event][-1])
            batch_feed_dict[self.placeholders['events_partition_idx_ph'][event]] = event_partition
            batch_feed_dict[self.placeholders['events_nodes_ph'][event]] = np.asarray(event_data, dtype=np.int32)
            batch_feed_dict[self.placeholders['events_nodes_type_ph'][event]] = np.asarray(event_data_type, dtype=np.int32)
            sampled_his_event, his_event_deltatime, has_neighbor, event_deltatime = self.sample_subevent_fromhis(batch_data[event], self.node_his_event, sub_events, events_time, self._his_type)
            batch_feed_dict[self.placeholders['events_deltatime_ph'][event]] = np.asarray(event_deltatime, dtype=np.float64)
            batch_feed_dict[self.placeholders['events_nodes_history_ph'][event]] = np.asarray(sampled_his_event, dtype=np.int32)
            batch_feed_dict[self.placeholders['events_nodes_history_deltatime_ph'][event]] = np.asarray(his_event_deltatime, dtype=np.float64)
            batch_feed_dict[self.placeholders['has_neighbor'][event]] = has_neighbor
            for node_type in range(self._type_num):
                for node in batch_data[event][node_type]:
                    self.node_his_event[node].append(batch_data[event][-1])
        return batch_feed_dict, epoch_flag

    def sample_negbatch_events_eval1(self, batch_data, neg_num):
        batch_data_neg_list = []
        for event in range(len(batch_data)):
            data_neg = [[[] for _ in range(self._type_num+3)] for _ in range(neg_num)]
            for neg in range(neg_num):
                # neg_type = random.choice(range(self._type_num))
                neg_type = 0
                for type in range(self._type_num):
                    if neg_type == type:
                        prenum = 0
                        for pretype in range(type):
                            prenum += self._num_node_type[pretype]
                        # data_neg[neg][type] = list(np.random.randint(prenum, prenum+self._num_node_type[type]-1, len(batch_data[event][type])))

                        replace_node = random.choice(batch_data[event][type])
                        for node in batch_data[event][type]:
                            if node == replace_node:
                                while True:
                                    neg_node = random.randint(prenum, prenum+self._num_node_type[type] - 1)
                                    if neg_node in batch_data[event][type]:
                                        continue
                                    else:
                                        data_neg[neg][type].append(neg_node)
                                        break
                            else:
                                data_neg[neg][type].append(node)

                    else:
                        data_neg[neg][type] = batch_data[event][type]
                data_neg[neg][-1] = batch_data[event][-1]
                data_neg[neg][-2] = batch_data[event][-2]
                data_neg[neg][-3] = batch_data[event][-3]
            batch_data_neg_list.append(data_neg)
        return batch_data_neg_list

    def get_feed_dict_evo(self, is_init):
        feed_dict_eval = {}
        eval_data, epoch_flag = self.test_data.next_batch()
        sub_events={**self._sub_events_train, **self._sub_events_valid, **self._sub_events_test}
        events_time = {**self._events_time_train, **self._events_time_valid, **self._events_time_test}
        feed_dict_eval[self.placeholders['nodes_embedding_ph']] = self.node_embedding_cur
        feed_dict_eval[self.placeholders['nodes_cellstates_ph']] = self.node_cellstates_cur
        feed_dict_eval[self.placeholders['is_init']] = is_init
        feed_dict_eval[self.placeholders['keep_prob']] = 1.0
        eval_data = eval_data[0]
        ###################record history event for each node#####################
        event_partition = np.zeros(self._num_node)
        event_data = []
        event_data_type = []
        for node_type in range(self._type_num):
            event_data.extend(eval_data[node_type])
            event_data_type.extend([node_type]*len(eval_data[node_type]))
            event_partition[eval_data[node_type]] = node_type+1
            # for node in eval_data[node_type]:
            #     self.node_his_event[node].append(eval_data[-1])
        feed_dict_eval[self.placeholders['event_partition_idx_eval_ph']] = event_partition
        feed_dict_eval[self.placeholders['event_nodes_eval_ph']] = np.asarray(event_data, dtype=np.int32)
        feed_dict_eval[self.placeholders['event_nodes_type_eval_ph']] = np.asarray(event_data_type, dtype=np.int32)
        sampled_his_event, his_event_deltatime, has_neighbor, event_deltatime = self.sample_subevent_fromhis(eval_data, self.node_his_event, sub_events, events_time, self._his_type)
        feed_dict_eval[self.placeholders['event_nodes_history_eval_ph']] = np.asarray(sampled_his_event, dtype=np.int32)
        feed_dict_eval[self.placeholders['event_nodes_history_deltatime_eval_ph']] = np.asarray(his_event_deltatime, dtype=np.float64)
        feed_dict_eval[self.placeholders['has_neighbor_eval']] = has_neighbor
        feed_dict_eval[self.placeholders['event_deltatime_eval_ph']] = np.asarray(event_deltatime, dtype=np.float64)
        #############################################
        # for node_type in range(self._type_num):
        #     for node in eval_data[node_type]:
        #         self.node_his_event[node].append(eval_data[-1])
        return feed_dict_eval, eval_data, epoch_flag

    def get_feed_dict_eval(self, eval_data_neg_list, start_idx, is_init):
        feed_dict_eval = {}
        sub_events={**self._sub_events_train, **self._sub_events_valid, **self._sub_events_test}
        events_time = {**self._events_time_train, **self._events_time_valid, **self._events_time_test}
        feed_dict_eval[self.placeholders['nodes_embedding_ph']] = self.node_embedding_cur
        feed_dict_eval[self.placeholders['nodes_cellstates_ph']] = self.node_cellstates_cur
        feed_dict_eval[self.placeholders['is_init']] = is_init
        feed_dict_eval[self.placeholders['keep_prob']] = self._keep
        eval_data_neg = eval_data_neg_list[start_idx]
        neg_flag = False
        if (start_idx+1 == len(eval_data_neg_list)):
            start_idx = 0
            neg_flag = True
        else:
            start_idx += 1
        ###################record history event for each node#####################
        event_data_neg = []
        event_data_neg_type = []
        for node_type in range(self._type_num):
            event_data_neg.extend(eval_data_neg[node_type])
            event_data_neg_type.extend([node_type]*len(eval_data_neg[node_type]))
        feed_dict_eval[self.placeholders['negevent_nodes_eval_ph']] = np.asarray(event_data_neg, dtype=np.int32)
        feed_dict_eval[self.placeholders['negevent_nodes_type_eval_ph']] = np.asarray(event_data_neg_type, dtype=np.int32)
        sampled_his_event_neg, his_event_deltatime_neg, has_neighbor_neg, _ = self.sample_subevent_fromhis(eval_data_neg, self.node_his_event, sub_events, events_time, self._his_type)
        feed_dict_eval[self.placeholders['negevent_nodes_history_eval_ph']] = np.asarray(sampled_his_event_neg, dtype=np.int32)
        feed_dict_eval[self.placeholders['negevent_nodes_history_deltatime_eval_ph']] = np.asarray(his_event_deltatime_neg, dtype=np.int32)
        feed_dict_eval[self.placeholders['has_neighbor_neg_eval']] = has_neighbor_neg
        #############################################
        return feed_dict_eval, start_idx, neg_flag

    def sample_negbatch_events_eval(self, batch_data, neg_type, replace_node):
        prenum = 0
        for pretype in range(neg_type):
            prenum += self._num_node_type[pretype]
        start_typeidx = prenum
        neg_size = self._num_node_type[neg_type] - 1
        data_neg = [[[] for _ in range(self._type_num+3)] for _ in range(neg_size)]
        for neg in range(neg_size):
            for type in range(self._type_num):
                if neg_type == type:
                    for node in batch_data[type]:
                        if replace_node == node:
                            if start_typeidx == replace_node:
                                start_typeidx += 1
                            data_neg[neg][type].append(start_typeidx)
                            start_typeidx += 1
                        else:
                            data_neg[neg][type].append(node)
                else:
                    data_neg[neg][type] = batch_data[type]
                data_neg[neg][-1] = batch_data[-1]
                data_neg[neg][-2] = batch_data[-2]
                data_neg[neg][-3] = batch_data[-3]
        return data_neg









