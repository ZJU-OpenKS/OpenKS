'''Citation
@inproceedings{rozemberczki2019gemsec,
                title={{GEMSEC: Graph Embedding with Self Clustering}},
                author={Rozemberczki, Benedek and Davies, Ryan and Sarkar, Rik and Sutton, Charles},
                booktitle={Proceedings of the 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2019},
                pages={65-72},
                year={2019},
                organization={ACM}
                }
'''

import random
import time
import numpy as np
import math
import tensorflow.compat.v1 as tf
import networkx as nx
from tqdm import tqdm

from openks.models.model import TFModel
from .layers import DeepWalker, Clustering, Regularization
from .utils import *

tf.disable_v2_behavior()


@TFModel.register("Model", "TFModel")
class Model(TFModel):
    """
    Abstract model class.
    """

    def __init__(self, config, graph):
        """
        initialize args and graph
        :param config: configuration
        :param graph: relation network
        """

        self.config = config
        self.graph = graph
        if self.config['walker'] == "first":
            self.walker = RandomWalker(self.graph, nx.nodes(graph), self.config['num_of_walks'],
                                       self.config['random_walk_length'])
            self.degrees, self.walks = self.walker.do_walks()
        else:
            self.walker = SecondOrderRandomWalker(self.graph, False, self.config['P'], self.config['Q'])
            self.walker.preprocess_transition_probs()
            self.walks, self.degrees = self.walker.simulate_walks(self.config['num_of_walks'],
                                                                  self.config['random_walk_length'])
        self.nodes = self.graph.nodes()
        del self.walker
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.config['num_of_walks'] * self.vocab_size
        self.build()

    def build(self):
        """
        model building
        :return:
        """

        pass

    def feed_dict_generator(self):
        """
        feed generator
        :return: feed dictionary
        """

        pass

    def train(self):
        """
        model train
        :return: model
        """

        pass


@TFModel.register("GEMSECWithRegularization", "TFModel")
class GEMSECWithRegularization(Model):
    """
    Regularized GEMSEC class.
    """

    def build(self):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.walker_layer = DeepWalker(self.config, self.vocab_size, self.degrees)
            self.cluster_layer = Clustering(self.config)
            self.regularizer_layer = Regularization(self.config)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer() + self.gamma * self.cluster_layer(
                self.walker_layer) + self.regularizer_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.config['initial_learning_rate'],
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.config['minimal_learning_rate'],
                                                               self.config['annealing_factor'])

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step=self.batch)

            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.config, self.graph)

    def feed_dict_generator(self, a_random_walk, step, gamma):
        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk, self.config['random_walk_length'],
                                             self.config['window_size'])

        batch_labels = batch_label_generator(a_random_walk, self.config['random_walk_length'],
                                             self.config['window_size'])

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.regularizer_layer.edge_indices_left: index_1,
                     self.regularizer_layer.edge_indices_right: index_2,
                     self.regularizer_layer.overlap: overlaps}

        return feed_dict

    def train(self):
        self.current_step = 0
        self.current_gamma = self.config['initial_gamma']
        self.log = log_setup(self.config)

        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.config['num_of_walks']):

                random.shuffle(list(self.nodes))
                self.optimization_time = 0
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.config['initial_gamma'],
                                                           self.current_gamma, self.true_step_size)
                    feed_dict = self.feed_dict_generator(self.walks[self.current_step - 1], self.current_step,
                                                         self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end - start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss / self.vocab_size
                self.final_embeddings = self.walker_layer.embedding_matrix.eval()
                if "GEMSEC" in self.config['model']:
                    self.c_means = self.cluster_layer.cluster_means.eval()
                    self.modularity_score, assignments = neural_modularity_calculator(self.graph, self.final_embeddings,
                                                                                      self.c_means)
                else:
                    self.modularity_score, assignments = classical_modularity_calculator(self.graph,
                                                                                         self.final_embeddings,
                                                                                         self.config)
                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time,
                                       self.modularity_score)
                tab_printer(self.log)
        if "GEMSEC" in self.config['model']:
            initiate_dump_gemsec(self.log, assignments, self.config, self.final_embeddings, self.c_means)
        else:
            initiate_dump_dw(self.log, assignments, self.config, self.final_embeddings)


@TFModel.register("GEMSEC", "TFModel")
class GEMSEC(GEMSECWithRegularization):
    """
    GEMSEC class.
    """

    def build(self):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.walker_layer = DeepWalker(self.config, self.vocab_size, self.degrees)
            self.cluster_layer = Clustering(self.config)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer() + self.gamma * self.cluster_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.config['initial_learning_rate'],
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.config['minimal_learning_rate'],
                                                               self.config['annealing_factor'])

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step=self.batch)

            self.init = tf.global_variables_initializer()

    def feed_dict_generator(self, a_random_walk, step, gamma):
        batch_inputs = batch_input_generator(a_random_walk, self.config['random_walk_length'],
                                             self.config['window_size'])

        batch_labels = batch_label_generator(a_random_walk, self.config['random_walk_length'],
                                             self.config['window_size'])

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step)}

        return feed_dict


@TFModel.register("DeepWalkWithRegularization", "TFModel")
class DeepWalkWithRegularization(GEMSECWithRegularization):
    """
    Regularized DeepWalk class.
    """

    def build(self):
        """
        Method to create the computational graph and initialize weights.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.walker_layer = DeepWalker(self.config, self.vocab_size, self.degrees)
            self.regularizer_layer = Regularization(self.config)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer() + self.regularizer_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.config['initial_learning_rate'],
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.config['minimal_learning_rate'],
                                                               self.config['annealing_factor'])

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step=self.batch)

            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.config, self.graph)

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate random walk features, left and right handside matrices, proper time index and overlap vector.
        """
        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk, self.config['random_walk_length'], self.config['window_size'])

        batch_labels = batch_label_generator(a_random_walk, self.config['random_walk_length'], self.config['window_size'])

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.regularizer_layer.edge_indices_left: index_1,
                     self.regularizer_layer.edge_indices_right: index_2,
                     self.regularizer_layer.overlap: overlaps}

        return feed_dict


@TFModel.register("DeepWalk", "TFModel")
class DeepWalk(GEMSECWithRegularization):
    """
    DeepWalk class.
    """

    def build(self):
        """
        Method to create the computational graph and initialize weights.
        """

        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.walker_layer = DeepWalker(self.config, self.vocab_size, self.degrees)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.config['initial_learning_rate'],
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.config['minimal_learning_rate'],
                                                               self.config['annealing_factor'])

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step=self.batch)

            self.init = tf.global_variables_initializer()

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate random walk features, gamma and proper time index.
        """

        batch_inputs = batch_input_generator(a_random_walk, self.config['random_walk_length'], self.config['window_size'])
        batch_labels = batch_label_generator(a_random_walk, self.config['random_walk_length'], self.config['window_size'])

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step)}

        return feed_dict
