import sys
import random
import numpy as np
import pickle as pkl
import math
import tensorflow as tf
from utils.inits import glorot_init, zeros_init
import sklearn.metrics as metrics
from models.basic_model import BasicModel
from utils.data_manager import *

class GnnEventModel_learnsuc(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'negative_ratio': 5,
            'table_size': 1e8,
            'neg_power': 0.75,
            # 'use_event_bias': True
        })
        return params

    def get_log_file(self, params):
        log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'learnsuc'
        return log_file

    def get_checkpoint_dir(self, params):
        checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'learnsuc'
        return checkpoint_dir

    def make_model(self):
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['loss'] = self.build_specific_graph_model()
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    def _create_placeholders(self):
            self.placeholders['events_nodes_type_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                        name='node_type%i_event%i_ph'%(node_type, event))
                                                        for node_type in range(self._type_num)]
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['negevents_nodes_type_ph'] = [[[tf.placeholder(tf.int32, shape=[None],
                                            name='node_type%i_event%i_neg%i_ph'%(node_type, event, neg))
                                            for node_type in range(self._type_num)]
                                            for event in range(self._eventnum_batch)]
                                            for neg in range(self._neg_num)]

    def _create_variables(self):
        cur_seed = random.getrandbits(32)
        self._node_embedding_st = [tf.get_variable('node_type%i_embedding_st'%(node_type),
                                shape=[self._num_node_type[node_type], self._h_dim], dtype=tf.float64, trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
                                for node_type in range(self._type_num)]

    def prepare_specific_graph_model(self):
        self._h_dim = self.params['hidden_size']
        self._type_num = self.params['node_type_numbers']
        self._eventnum_batch = self.params['batch_event_numbers']
        self._neg_num = self.params['negative_ratio']
        self._num_node_type = self.params['n_nodes_pertype']

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
        self.weights['type_weights'] = [tf.Variable(glorot_init([self._h_dim, self._h_dim],
                                        name='gnn_type_weights_typ%i' % node_type))
                                        for node_type in range(self._type_num)]

        self.weights['type_weights_scalar'] = [tf.Variable(1.0, dtype=tf.float64, trainable=True) for node_type in range(self._type_num)]
        # self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._eventclass]))
        self.weights['event_weights'] = tf.Variable(glorot_init([self._h_dim, self._h_dim]))

    def build_specific_graph_model(self):
        event_states_list = []
        neg_event_states_list = []
        for event_id in range(self._eventnum_batch):
            event_states = tf.zeros([self._h_dim], dtype=tf.float64)
            neg_event_states = [tf.zeros([self._h_dim], dtype=tf.float64) for _ in range(self._neg_num)]
            for node_type in range(self._type_num):
                states = tf.nn.embedding_lookup(self._node_embedding_st[node_type],
                                                self.placeholders['events_nodes_type_ph'][event_id][node_type])
                states_aggregated = tf.reduce_mean(states, axis=0, keepdims=True)
                # event_states += tf.matmul(states_aggregated, self.weights['type_weights'][node_type])
                event_states += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], states_aggregated)
            # event_states = self.activation_fun(tf.matmul(event_states, self.weights['event_weights'])) + event_states
            for neg in range(self._neg_num):
                for node_type in range(self._type_num):
                    neg_states = tf.nn.embedding_lookup(self._node_embedding_st[node_type],
                                                self.placeholders['negevents_nodes_type_ph'][neg][event_id][node_type])
                    neg_states_aggregated = tf.reduce_mean(neg_states, axis=0, keepdims=True)
                    # neg_event_states[neg] += tf.matmul(neg_states_aggregated, self.weights['type_weights'][node_type])
                    neg_event_states[neg] += tf.scalar_mul(self.weights['type_weights_scalar'][node_type], neg_states_aggregated)
                # neg_event_states[neg] = self.activation_fun(tf.matmul(neg_event_states[neg], self.weights['event_weights'])) + neg_event_states[neg]
            neg_event_states = tf.stack(neg_event_states)
            event_states_list.append(tf.squeeze(event_states))
            neg_event_states_list.append(tf.squeeze(neg_event_states))
            # neg_event_states_list.append([tf.squeeze(neg_event_states[neg]) for neg in range(self._neg_num)])
        event_scores = tf.stack(event_states_list)
        neg_event_scores = tf.stack(neg_event_states_list)
        event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(event_scores, event_scores), axis=1))
        neg_event_scores_norms = tf.sqrt(tf.reduce_sum(tf.multiply(neg_event_scores, neg_event_scores), axis=2))

        event_losses = tf.log(tf.tanh(event_scores_norms/2))
        neg_event_losses = tf.log(tf.tanh(tf.reciprocal(neg_event_scores_norms)/2))
        neg_event_losses = tf.reduce_sum(neg_event_losses, axis=1)
        losses = event_losses + neg_event_losses
        loss_mean = -tf.reduce_mean(losses)
        return loss_mean

    def sample_negbatch_events(self, batch_data):
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


    def get_batch_feed_dict(self, state):
        batch_feed_dict = {}
        if state == 'train':
            batch_data, epoch_flag = self.train_data.next_batch()
        elif state == 'valid':
            batch_data, epoch_flag = self.valid_data.next_batch()
        else:
            print('state wrong')
        batch_data_neg_list = self.sample_negbatch_events(batch_data)
        for event in range(self._eventnum_batch):
            for node_type in range(self._type_num):
                batch_feed_dict[self.placeholders['events_nodes_type_ph'][event][node_type]] = np.asarray(batch_data[event][node_type], dtype=np.int32)
                for neg in range(self.params['negative_ratio']):
                    batch_feed_dict[self.placeholders['negevents_nodes_type_ph'][neg][event][node_type]] = \
                        np.asarray(batch_data_neg_list[neg][event][node_type], dtype=np.int32)
        return batch_feed_dict, epoch_flag

    def sample_negtest_events(self, test_data):
        test_data_neg = [[] for _ in range(self._type_num)]
        for type in range(self._type_num):
            while (len(test_data_neg[type]) < len(test_data[type])):
                neg_node = random.randint(0, self._num_node_type[type]-1)
                if neg_node in test_data_neg[type]:
                    continue
                test_data_neg[type].append(neg_node)
        return test_data_neg

    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        print('batches'+str(train_batches_num))
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []
                epoch_flag = False
                print('start epoch %i'%(epoch))
                while not epoch_flag:
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train')
                    fetches = [self.ops['loss'], self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                    cost, step, lr, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)
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
                    if step == 1 or step % self.params['eval_point'] == 0:
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
                            log_out.write('model saved to {}'.format(ckpt_path))
                            print('model saved to {}'.format(ckpt_path))
                            # save_trained_embeddings(self.node_embedding_cur, self.params['embedding_out_base'])
                            # print('save embeddings to:' + self.params['embedding_out_base'])
                            sys.stdout.flush()
                    if epoch-best_epoch >= self.params['patience']:
                        log_out.write('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                        print('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                        break
            log_out.write('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            print('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            print('start test')

    def validation(self):
        valid_batches_num = self.valid_data.get_batch_num()
        valid_loss = []
        epoch_flag = False
        print('valid batches' + str(valid_batches_num))
        while not epoch_flag:
            fetches = self.ops['loss']
            batch_feed_dict, epoch_flag = self.get_batch_feed_dict('valid')
            cost = self.sess.run(fetches, feed_dict=batch_feed_dict)
            if np.isnan(cost):
                print('Evaluation loss Nan!')
                sys.exit(1)
            valid_loss.append(cost)
        return np.mean(valid_loss)

    def test(self):
        self.test_data.batch_size = 1
        test_batches_num = self.test_data.get_batch_num()
        print('test nums:'+str(test_batches_num))
        epoch_flag = False
        val_preds = []
        labels = []
        self.node_embedding_testcur = [None for _ in range(self._type_num)]
        for node_type in range(self._type_num):
            self.node_embedding_testcur[node_type] = self._node_embedding_st[node_type].eval(session=self.sess)
        # with open('/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/test_neg.pkl', 'rb') as test_neg_root:
        #     test_data_neg_list = pkl.load(test_neg_root)
        # num = 0
        while not epoch_flag:
            test_data, epoch_flag = self.test_data.next_batch()
            test_data = test_data[0]
            test_data_neg = self.sample_negtest_events(test_data)
            # test_data_neg = test_data_neg_list[num]
            # num += 1
            event_embedding = np.zeros(self._h_dim)
            negevent_embedding = np.zeros(self._h_dim)
            for node_type in range(self._type_num):
                ids = test_data[node_type]
                negids = test_data_neg[node_type]
                event_embedding += np.mean(self.node_embedding_testcur[node_type][ids], axis=0)
                negevent_embedding += np.mean(self.node_embedding_testcur[node_type][negids], axis=0)
            r = np.tanh(np.sqrt(np.sum(np.square(event_embedding))) / 2)
            negr = np.tanh(np.sqrt(np.sum(np.square(negevent_embedding))) / 2)
            val_preds.append(r)
            val_preds.append(negr)
            labels.append(1)
            labels.append(0)
        # precision = metrics.precision_score(labels, val_preds, average=None)
        # recall = metrics.recall_score(labels, val_preds, average=None)
        # f1 = metrics.f1_score(labels, val_preds, average=None)
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        # print('precision:%f, recall:%f, f1:%f' % precision, recall, f1)
        print('mae:%f, rmse:%f' % (mae, rmse))
        # print(num)


