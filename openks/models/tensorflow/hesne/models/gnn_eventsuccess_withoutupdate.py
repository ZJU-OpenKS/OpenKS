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

class GnnEventModel_withoutupdate(BasicModel):
    def __init__(self, args):
        super().__init__(args)

    def default_params(cls):
        params = dict(super().default_params())
        return params

    def get_log_file(self):
        log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'withoutupdate.log'
        return log_file

    def get_checkpoint_dir(self):
        checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                    + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                    + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob']) + 'withoutupdate'
        return checkpoint_dir

    def make_model(self):
        with tf.variable_scope('graph_model'):
            self.prepare_specific_graph_model()
            self.ops['loss'], self.ops['pred'], self.ops['neg_pred'] = self.build_specific_graph_model()
            self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict'] = self.build_specific_eval_graph_model()
            self.ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        print('model build')

    def _create_placeholders(self):
            self.placeholders['events_nodes_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event%i_ph'%event)
                                                        for event in range(self._eventnum_batch)]

            self.placeholders['events_nodes_type_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event%i_type_ph'%event)
                                                        for event in range(self._eventnum_batch)]


            self.placeholders['events_partition_idx_ph'] = [tf.placeholder(tf.int32, shape=[None],
                                                        name='event%i_partition_idx_ph'%event)
                                                        for event in range(self._eventnum_batch)]


            self.placeholders['negevents_nodes_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                    name='event%i_neg%i_ph'%(event, neg))
                                                    for neg in range(self._neg_num)]
                                                    for event in range(self._eventnum_batch)]

            self.placeholders['negevents_nodes_type_ph'] = [[tf.placeholder(tf.int32, shape=[None],
                                                    name='event%i_neg%i_type_ph'%(event, neg))
                                                    for neg in range(self._neg_num)]
                                                    for event in range(self._eventnum_batch)]

###########test placeholder###########################
            self.placeholders['event_nodes_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='event_eval_ph')

            self.placeholders['event_nodes_type_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='event_type_eval_ph')


            self.placeholders['event_partition_idx_eval_ph'] = tf.placeholder(tf.int32, shape=[None],
                                                            name='event_partition_idx_eval_ph')

            self.placeholders['negevent_nodes_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='negevent_eval_ph')

            self.placeholders['negevent_nodes_type_eval_ph'] = tf.placeholder(tf.int32, shape=[None], name='negevent_type_eval_ph')

######################################################

    def _create_variables(self):
        cur_seed = random.getrandbits(32)
        self._embedding_init = tf.get_variable('nodes_embedding_init', shape=[self._num_node, self._h_dim],
                                                dtype=tf.float64, trainable=True,
                                                # initializer=tf.random_uniform_initializer(-1, 1, seed=cur_seed))
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

        self.weights['type_weights_scalar'] = tf.Variable(tf.ones([self._type_num], dtype=tf.float64), trainable=True)


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
        self._keep = self.params['keep_prob']
        self._istraining = self.params['is_training']
        self._aggregator_type = self.params['aggregator_type'].lower()

        self._create_placeholders()
        self._create_variables()

    def build_specific_graph_model(self):
        event_pred_list = []
        neg_event_pred_list = []
        self.triangularize_layer = Triangularize()
        for event_id in range(self._eventnum_batch):
            neg_event_states_stacked = [None for _ in range(self._neg_num)]
            dy_states = tf.nn.embedding_lookup(self._embedding_init, self.placeholders['events_nodes_ph'][event_id])
            _, _, type_count = tf.unique_with_counts(self.placeholders['events_nodes_type_ph'][event_id])
            event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'],
                                                   self.placeholders['events_nodes_type_ph'][event_id])
            type_count = tf.cast(type_count, tf.float64)
            count_weights = tf.nn.embedding_lookup(tf.reciprocal(type_count),
                                                   self.placeholders['events_nodes_type_ph'][event_id])
            event_weights = tf.multiply(event_weights, count_weights)
            # event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['events_nodes_type_ph'][event_id])
            event_states = dy_states*tf.expand_dims(event_weights, 1)
            for neg in range(self._neg_num):
                neg_dy_states = tf.nn.embedding_lookup(self._embedding_init, self.placeholders['negevents_nodes_ph'][event_id][neg])
                _, _, neg_type_count = tf.unique_with_counts(
                    self.placeholders['negevents_nodes_type_ph'][event_id][neg])
                neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'],
                                                           self.placeholders['negevents_nodes_type_ph'][event_id][neg])
                neg_type_count = tf.cast(neg_type_count, tf.float64)
                neg_count_weights = tf.nn.embedding_lookup(tf.reciprocal(neg_type_count),
                                                           self.placeholders['negevents_nodes_type_ph'][event_id][neg])
                neg_event_weights = tf.multiply(neg_event_weights, neg_count_weights)
                # neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['negevents_nodes_type_ph'][event_id][neg])
                neg_event_states = neg_dy_states*tf.expand_dims(neg_event_weights, 1)
                neg_event_states_stacked[neg] = neg_event_states
        # ###pairwise layer to predict
            event_scores = tf.expand_dims(event_states, 0)
            neg_event_scores = tf.stack(neg_event_states_stacked)
            event_scores_h = tf.matmul(event_scores, event_scores, transpose_b=True)
            event_scores_h = self.triangularize_layer(event_scores_h)
            event_scores_h = tf.layers.flatten(event_scores_h)
            y_pred = tf.reduce_sum(event_scores_h, 1) ##change sum to mean
            neg_event_scores_h = tf.matmul(neg_event_scores, neg_event_scores, transpose_b=True)
            neg_event_scores_h = self.triangularize_layer(neg_event_scores_h)
            neg_event_scores_h = tf.layers.flatten(neg_event_scores_h)
            neg_y_pred = tf.reduce_sum(neg_event_scores_h, 1) ##change sum to mean
            event_pred_list.append(y_pred)
            neg_event_pred_list.append(neg_y_pred)
        pred = tf.squeeze(tf.stack(event_pred_list))
        neg_pred = tf.squeeze(tf.stack(neg_event_pred_list))
        event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pred), logits=pred)
        neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_pred), logits=neg_pred)
        loss_mean = (tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses)) / self._eventnum_batch
        predict = tf.sigmoid(pred)
        neg_predict = tf.sigmoid(neg_pred)
        return loss_mean, predict, neg_predict

    def build_specific_eval_graph_model(self):
        dy_states = tf.nn.embedding_lookup(self._embedding_init, self.placeholders['event_nodes_eval_ph'])
        _, _, type_count = tf.unique_with_counts(self.placeholders['event_nodes_type_eval_ph'])
        event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'],
                                               self.placeholders['event_nodes_type_eval_ph'])
        type_count = tf.cast(type_count, tf.float64)
        count_weights = tf.nn.embedding_lookup(tf.reciprocal(type_count),
                                               self.placeholders['event_nodes_type_eval_ph'])
        event_weights = tf.multiply(event_weights, count_weights)
        # event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['event_nodes_type_eval_ph'])
        event_states = dy_states * tf.expand_dims(event_weights, 1)
        neg_dy_states = tf.nn.embedding_lookup(self._embedding_init, self.placeholders['negevent_nodes_eval_ph'])
        _, _, neg_type_count = tf.unique_with_counts(self.placeholders['negevent_nodes_type_eval_ph'])
        neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'],
                                                   self.placeholders['negevent_nodes_type_eval_ph'])
        neg_type_count = tf.cast(neg_type_count, tf.float64)
        neg_count_weights = tf.nn.embedding_lookup(tf.reciprocal(neg_type_count),
                                                   self.placeholders['negevent_nodes_type_eval_ph'])
        neg_event_weights = tf.multiply(neg_event_weights, neg_count_weights)
        # neg_event_weights = tf.nn.embedding_lookup(self.weights['type_weights_scalar'], self.placeholders['negevent_nodes_type_eval_ph'])
        neg_event_states = neg_dy_states * tf.expand_dims(neg_event_weights, 1)
        ###pairwise layer to predict
        event_scores = event_states
        neg_event_scores = neg_event_states
        event_scores = tf.expand_dims(event_scores, 0)
        neg_event_scores = tf.expand_dims(neg_event_scores, 0)
        event_scores_h = tf.matmul(event_scores, event_scores, transpose_b=True)
        event_scores_h = self.triangularize_layer(event_scores_h)
        event_scores_h = tf.layers.flatten(event_scores_h)
        y_pred = tf.reduce_sum(event_scores_h, 1, keepdims=True) ##change sum to mean
        neg_event_scores_h = tf.matmul(neg_event_scores, neg_event_scores, transpose_b=True)
        neg_event_scores_h = self.triangularize_layer(neg_event_scores_h)
        neg_event_scores_h = tf.layers.flatten(neg_event_scores_h)
        neg_y_pred = tf.reduce_sum(neg_event_scores_h, 1, keepdims=True) ##change sum to mean
        event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)
        neg_event_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_y_pred), logits=neg_y_pred)
        loss_mean = tf.reduce_sum(event_losses) + tf.reduce_sum(neg_event_losses)
        predict = tf.sigmoid(tf.squeeze(y_pred))
        neg_predict = tf.sigmoid(tf.squeeze(neg_y_pred))
        return loss_mean, predict, neg_predict


    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        print('batches'+str(train_batches_num))
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                self.train_data.shuffle()
                epoch_loss = []
                epoch_flag = False
                print('start epoch %i'%(epoch))
                while not epoch_flag:
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train')
                    fetches = [self.ops['loss'], self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                    cost, step, lr, _ = self.sess.run(fetches, feed_dict=batch_feed_dict)
                    epoch_loss.append(cost)
                    if np.isnan(cost):
                        log_out.write('Train ' + str(epoch) + ':Nan error!\n')
                        print('Train ' + str(epoch) + ':Nan error!')
                        return
                    if step == 1 or step % (self.params['decay_step']/10) == 0:
                        avgc = np.mean(epoch_loss)
                        log_out.write('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}\tavgc:{:.6f}\n'.format(epoch, step, lr, cost, avgc))
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}\tavgc:{:.6f}'.format(epoch, step, lr, cost, avgc))
                        sys.stdout.flush()
                        log_out.flush()
                print('start valid')
                valid_loss = self.validation(log_out)
                log_out.write('Evaluation loss after step {}: {:.6f}\n'.format(step, valid_loss))
                print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                print('start test')
                self.test(log_out)
                if valid_loss < best_loss:
                    best_epoch = epoch
                    best_loss = valid_loss
                    ckpt_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
                    self.saver.save(self.sess, ckpt_path, global_step=step)
                    print('model saved to {}'.format(ckpt_path))
                    log_out.write('model saved to {}\n'.format(ckpt_path))
                    sys.stdout.flush()
                if epoch-best_epoch >= self.params['patience']:
                    log_out.write('Stopping training after %i epochs without improvement on validation.\n' % self.params['patience'])
                    print('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                    break
            log_out.write('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}\n'.format(best_epoch, best_loss))
            print('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))


    def validation(self, log_out):
        epoch_flag = False
        valid_loss = []
        valid_batches_num = self.valid_data.get_batch_num()
        print('valid nums:' + str(valid_batches_num))
        log_out.write('valid nums:\n' + str(valid_batches_num))
        labels = []
        val_preds = []
        self.valid_data.shuffle()
        while not epoch_flag:
            fetches = [self.ops['loss'], self.ops['pred'], self.ops['neg_pred']]
            feed_dict_valid, epoch_flag = self.get_batch_feed_dict('valid')
            cost, pred, neg_pred = self.sess.run(fetches, feed_dict=feed_dict_valid)
            print(cost)
            log_out.write('cost:{:.6f}\n'.format(cost))
            valid_loss.append(cost)
            val_preds.extend(list(pred))
            val_preds.extend(list(neg_pred[:,0]))
            labels.extend([1 for _ in range(self._eventnum_batch)])
            labels.extend([0 for _ in range(self._eventnum_batch)])
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        fpr, tpr, thresholds = metrics.roc_curve(labels, val_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(labels, val_preds)
        # f1 = metrics.f1_score(labels, val_preds)
        print('mae:%f, rmse:%f, auc:%f, ap:%f' % (mae, rmse, auc, ap))
        log_out.write('mae:{:.6f}\trmse:{:.6f}\tauc:{:.6f}\tap:{:.6f}\n'.format(mae, rmse, auc, ap))
        return np.mean(valid_loss)


    def test(self, log_out):
        self.test_data.batch_size = 1
        test_batches_num = self.test_data.get_batch_num()
        print('test nums:'+str(test_batches_num))
        log_out.write('test nums:\n' + str(test_batches_num))
        epoch_flag = False
        val_preds = []
        labels = []
        test_loss = []
        self.test_data.shuffle()
        while not epoch_flag:
            fetches = [self.ops['loss_eval'], self.ops['predict'], self.ops['neg_predict']]
            feed_dict_test, epoch_flag = self.get_feed_dict_eval()
            cost, predict, neg_predict = self.sess.run(fetches, feed_dict=feed_dict_test)
            val_preds.append(predict)
            val_preds.append(neg_predict)
            test_loss.append(cost)
            labels.append(1)
            labels.append(0)
        # precision = metrics.precision_score(labels, val_preds, average=None)
        # recall = metrics.recall_score(labels, val_preds, average=None)
        # f1 = metrics.f1_score(labels, val_preds, average=None)
        print('label')
        print(labels[:100])
        print('pred')
        print(val_preds[:100])
        mae = metrics.mean_absolute_error(labels, val_preds)
        rmse = np.sqrt(metrics.mean_squared_error(labels, val_preds))
        fpr, tpr, thresholds = metrics.roc_curve(labels, val_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auc = metrics.roc_auc_score(labels, val_preds)
        ap = metrics.average_precision_score(labels, val_preds)
        print('mae:%f, rmse:%f, auc:%f, ap:%f' % (mae, rmse, auc, ap))
        log_out.write('mae:{:.6f}\trmse:{:.6f}\tauc:{:.6f}\tap:{:.6f}\n'.format(mae, rmse, auc, ap))
        print('test cost%f'%(np.mean(test_loss)))
        log_out.write('test cost:{:.6f}\n'.format(np.mean(test_loss)))

    def sample_negbatch_events(self, batch_data, neg_num):
        batch_data_neg_list = []
        for event in range(len(batch_data)):
            data_neg = [[[] for _ in range(self._type_num+3)] for _ in range(neg_num)]
            for neg in range(neg_num):
                neg_type = random.choice(range(self._type_num))
                # neg_type = 0
                for type in range(self._type_num):
                    if neg_type == type:
                        replace_node = random.choice(batch_data[event][type])
                        for node in batch_data[event][type]:
                            # if replace_node == node:
                            if True:
                                while True:
                                    prenum = 0
                                    for pretype in range(type):
                                        prenum += self._num_node_type[pretype]
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

    def get_batch_feed_dict(self, state):
        batch_feed_dict = {}
        if state == 'train':
            batch_data, epoch_flag = self.train_data.next_batch()
        elif state == 'valid':
            batch_data, epoch_flag = self.valid_data.next_batch()
        else:
            print('state wrong')
        batch_data_neg = self.sample_negbatch_events(batch_data, self._neg_num)
        for event in range(self._eventnum_batch):
            ###############record history event for each node###################
            event_data = []
            event_data_type = []
            event_data_neg = [[] for _ in range(self._neg_num)]
            event_data_neg_type = [[] for _ in range(self._neg_num)]
            for node_type in range(self._type_num):
                event_data.extend(batch_data[event][node_type])
                event_data_type.extend([node_type]*len(batch_data[event][node_type]))
                for neg in range(self._neg_num):
                    event_data_neg[neg].extend(batch_data_neg[event][neg][node_type])
                    event_data_neg_type[neg].extend([node_type]*len(batch_data_neg[event][neg][node_type]))
            batch_feed_dict[self.placeholders['events_nodes_ph'][event]] = np.asarray(event_data, dtype=np.int32)
            batch_feed_dict[self.placeholders['events_nodes_type_ph'][event]] = np.asarray(event_data_type, dtype=np.int32)
            ####
            for neg in range(self._neg_num):
                batch_feed_dict[self.placeholders['negevents_nodes_ph'][event][neg]] = \
                        np.asarray(event_data_neg[neg], dtype=np.int32)
                batch_feed_dict[self.placeholders['negevents_nodes_type_ph'][event][neg]] = \
                        np.asarray(event_data_neg_type[neg], dtype=np.int32)
        #############################################
        return batch_feed_dict, epoch_flag

    def get_feed_dict_eval(self):
        feed_dict_eval = {}
        eval_data, epoch_flag = self.test_data.next_batch()
        eval_data_neg = self.sample_negbatch_events(eval_data, 1)[0][0]
        eval_data = eval_data[0]
        ###################record history event for each node#####################
        event_data = []
        event_data_type = []
        event_data_neg = []
        event_data_neg_type = []
        for node_type in range(self._type_num):
            event_data.extend(eval_data[node_type])
            event_data_type.extend([node_type]*len(eval_data[node_type]))
            event_data_neg.extend(eval_data_neg[node_type])
            event_data_neg_type.extend([node_type]*len(eval_data_neg[node_type]))
        feed_dict_eval[self.placeholders['event_nodes_eval_ph']] = np.asarray(event_data, dtype=np.int32)
        feed_dict_eval[self.placeholders['event_nodes_type_eval_ph']] = np.asarray(event_data_type, dtype=np.int32)
        feed_dict_eval[self.placeholders['negevent_nodes_eval_ph']] = np.asarray(event_data_neg, dtype=np.int32)
        feed_dict_eval[self.placeholders['negevent_nodes_type_eval_ph']] = np.asarray(event_data_neg_type, dtype=np.int32)
        return feed_dict_eval, epoch_flag








