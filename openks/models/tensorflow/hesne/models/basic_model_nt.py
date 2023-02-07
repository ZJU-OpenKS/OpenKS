from typing import Tuple, List, Any, Sequence
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import sys, os
import json
from utils.data_manager import *
from models.hegraph import HeGraph
import time

class BasicModel_nt(object):
    @classmethod
    def default_params(cls):
        return{
            'num_epochs': 20,
            'hidden_size': 128,
            'learning_rate': 0.0005,
            'clamp_gradient_norm': 5.0,
            "negative_ratio": 5,
            'eval_point': 1e3,
            'decay_step': 5e2,
            'decay': 0.98,
            'patience': 5,
            'node_type_numbers': 3,
            'batch_event_numbers': 200,
            # 'use_propagation_attention': False,
            'graph_rnn_cell': 'lstm',
            'graph_rnn_activation': 'tanh',
            'keep_prob': 0.8,
            'use_different_cell': True,
            'is_training': True
        }

    def __init__(self, args):
        self.args = args
        data_dir = args['--data_dir']
        log_dir = args.get('--log_dir')
        save_dir = args.get('--save_dir')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        params = self.default_params()
        config_file = args.get('--config_file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--conifg')
        if config is not None:
            params.update((json.loads(config)))
        self.params = params

        self.log_file = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                        + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                        + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob'])+'_learnsuc'

        self.checkpoint_dir = 'h_' + str(self.params['hidden_size']) + '_b_' + str(self.params['batch_event_numbers']) \
                              + '_l_' + str(self.params['learning_rate']) + '_d_' + str(self.params['decay']) + '_ds_' \
                              + str(self.params['decay_step']) + '_k_' + str(self.params['keep_prob'])+'_learnsuc'
        self.log_file = os.path.join(log_dir, self.log_file)
        self.checkpoint_dir = os.path.join(save_dir, self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

# load train, valid, test data
        save_event_data(data_dir, self.params['batch_event_numbers'])
        self.train_data = load_event_data(os.path.join(data_dir, 'event_traindata.txt'))
        self.valid_data = load_event_data(os.path.join(data_dir, 'event_validdata.txt'))
        self.test_data = load_event_data(os.path.join(data_dir, 'event_testdata.txt'))
        print('load data done')
# build hegraph based on loaded data
        self.hegraph = HeGraph(self.params, self.train_data, self.valid_data, self.test_data)
        self.hegraph.build_hegraph()
        self.params['n_nodes_pertype'] = self.hegraph.get_totalnum_pertype()
        print('hegraph built')
        print(self.params['n_nodes_pertype'])
        self.train_data = BatchData(self.params, self.train_data)
        self.valid_data = BatchData(self.params, self.valid_data)
        self.test_data = BatchData(self.params, self.test_data)

# tensorflow config and graph build
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            # Restore/initialize variables:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            init_from = args.get('--init_from')
            if init_from is not None:
                ckpt = tf.train.get_checkpoint_state(init_from)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('init from:'+init_from)
            else:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                print('Randomly initialize model')

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def build_specific_graph_model(self, state) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")


    def make_train_step(self):
        self.ops['lr'] = tf.maximum(1e-5, tf.train.exponential_decay(self.params['learning_rate'], \
                            self.ops['global_step'], self.params['decay_step'], self.params['decay'], staircase=True))
        optimizer = tf.train.AdamOptimizer(self.ops['lr'])
        # grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'])
        print(tf.trainable_variables())
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_op'] = optimizer.apply_gradients(clipped_grads, global_step=self.ops['global_step'])
        print('make train step finished')

    def train(self):
        best_loss = 100.0
        best_epoch = -1
        train_batches_num = self.train_data.get_batch_num()
        print('batches'+str(train_batches_num))
        with open(self.log_file, 'w') as log_out:
            for epoch in range(self.params['num_epochs']):
                epoch_loss = []
                cost_test_list_train = []
                cost_test_list1_train = []
                epoch_flag = False
                print('start epoch %i'%(epoch))
                while not epoch_flag:
                    batch_feed_dict, epoch_flag = self.get_batch_feed_dict('train')
                    fetches = [self.ops['loss'],self.ops['loss_test'],self.ops['loss_test1'], self.ops['embedding'],\
                                   self.ops['global_step'], self.ops['lr'], self.ops['train_op']]
                    cost, cost_test, cost_test1, self.node_embedding_cur, step, lr, _ = \
                            self.sess.run(fetches, feed_dict=batch_feed_dict)
                    epoch_loss.append(cost)
                    cost_test_list_train.append(cost_test)
                    cost_test_list1_train.append(cost_test1)
                    if np.isnan(cost):
                        log_out.write('Train ' + str(epoch) + ':Nan error!')
                        print('Train ' + str(epoch) + ':Nan error!')
                        return
                    if step == 1 or step % (self.params['decay_step']/10) == 0:
                        avgc = np.mean(epoch_loss)
                        log_out.write('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        print(cost)
                        print('cost test'+str(np.mean(cost_test_list_train)))
                        print('cost test1' + str(np.mean(cost_test_list1_train)))
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tcost:{:.6f}'.format(epoch, step, lr, avgc))
                        sys.stdout.flush()
                    if step == 1 or step % self.params['eval_point'] == 0:
                        print('start valid')
                        valid_loss = self.validation()
                        log_out.write('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                        # self.test()
                        if valid_loss < best_loss:
                            best_epoch = epoch
                            best_loss = valid_loss
                            ckpt_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
                            self.saver.save(self.sess, ckpt_path, global_step=step)
                            log_out.write('model saved to {}'.format(ckpt_path))
                            print('model saved to {}'.format(ckpt_path))
                            save_trained_embeddings(self.node_embedding_cur, self.params['embedding_out_nt'])
                            print('save embeddings to:' + self.params['embedding_out_nt'])
                            sys.stdout.flush()
                        # epoch_time = time.time() - total_time_start
                    if epoch-best_epoch >= self.params['patience']:
                        log_out.write('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                        print('Stopping training after %i epochs without improvement on validation.' % self.params['patience'])
                        break
            log_out.write('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            print('Best evaluation loss appears in epoch {}. Lowest loss: {:.6f}'.format(best_epoch, best_loss))
            print('start test')
            # self.test()

    def validation(self):
        valid_batches_num = self.valid_data.get_batch_num()
        valid_loss = []
        cost_test_list = []
        cost_test_list1 = []
        epoch_flag = False
        print('valid batches' + str(valid_batches_num))
        while not epoch_flag:
            fetches = [self.ops['loss'], self.ops['loss_test'], self.ops['loss_test1']]
            batch_feed_dict, epoch_flag = self.get_batch_feed_dict('valid')
            cost, cost_test, cost_test1 = self.sess.run(fetches, feed_dict=batch_feed_dict)
            if np.isnan(cost):
                print('Evaluation loss Nan!')
                sys.exit(1)
            valid_loss.append(cost)
            cost_test_list.append(cost_test)
            cost_test_list1.append(cost_test1)
        print('valid cost test' + str(np.mean(cost_test_list)))
        print('valid cost test' + str(np.mean(cost_test_list1)))
        return np.mean(valid_loss)


