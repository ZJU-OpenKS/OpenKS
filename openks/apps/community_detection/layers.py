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

import tensorflow.compat.v1 as tf
import math
import numpy as np

tf.disable_v2_behavior()


class DeepWalker:
    """
    DeepWalk embedding layer class
    """

    def __init__(self, config, vocab_size, degrees):
        """
        Initialization of the layer with proper matrices and biases.
        The input variables are also initialized here.
        :param config: configuration
        :param vocab_size: the size of vocabulary
        :param degrees: the number of degrees
        """

        self.config = config
        self.vocab_size = vocab_size
        self.degrees = degrees
        self.train_labels = tf.placeholder(tf.int64, shape=[None, self.config['window_size']])

        self.train_inputs = tf.placeholder(tf.int64, shape=[None])

        self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.config['dimensions']],
                                                              -0.1 / self.config['dimensions'],
                                                              0.1 / self.config['dimensions']))

        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.config['dimensions']],
                                                           stddev=1.0 / math.sqrt(self.config['dimensions'])))

        self.nce_biases = tf.Variable(
            tf.random_uniform([self.vocab_size], -0.1 / self.config['dimensions'], 0.1 / self.config['dimensions']))

    def __call__(self):
        """
        Calculating the embedding cost with NCE and returning it.
        :return: embedding cost
        """

        self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
        self.input_ones = tf.ones_like(self.train_labels)
        self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs, [-1, 1])), [-1])
        self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix, self.train_inputs_flat, max_norm=1)

        self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes=self.train_labels_flat,
                                                             num_true=1,
                                                             num_sampled=self.config['negative_sample_number'],
                                                             unique=True,
                                                             range_max=self.vocab_size,
                                                             distortion=self.config['distortion'],
                                                             unigrams=self.degrees)

        self.embedding_losses = tf.nn.sampled_softmax_loss(weights=self.nce_weights,
                                                           biases=self.nce_biases,
                                                           labels=self.train_labels_flat,
                                                           inputs=self.embedding_partial,
                                                           num_true=1,
                                                           num_sampled=self.config['negative_sample_number'],
                                                           num_classes=self.vocab_size,
                                                           sampled_values=self.sampler)

        return tf.reduce_mean(self.embedding_losses)


class Clustering:
    """
    Latent space clustering class.
    """

    def __init__(self, config):
        """
        Initializing the cluster center matrix.
        :param config: configuration
        """

        self.config = config
        self.cluster_means = tf.Variable(tf.random_uniform([self.config['cluster_number'], self.config['dimensions']],
                                                           -0.1 / self.config['dimensions'],
                                                           0.1 / self.config['dimensions']))

    def __call__(self, Walker):
        """
        Calculating the clustering cost.
        :param Walker: walker
        :return: clustering cost
        """

        self.clustering_differences = tf.expand_dims(Walker.embedding_partial, 1) - self.cluster_means
        self.cluster_distances = tf.norm(self.clustering_differences, ord=2, axis=2)
        self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis=1)
        return tf.reduce_mean(self.to_be_averaged)


class Regularization:
    """
    Smoothness regularization class.
    """

    def __init__(self, config):
        """
        Initializing the indexing variables and the weight vector.
        :param config: configuration
        """

        self.config = config
        self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])
        self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
        self.overlap = tf.placeholder(tf.float32, shape=[None, 1])

    def __call__(self, Walker):
        """
        Calculating the regularization cost.
        :param Walker: walker
        :return: regularization cost
        """

        self.left_features = tf.nn.embedding_lookup(Walker.embedding_partial, self.edge_indices_left, max_norm=1)
        self.right_features = tf.nn.embedding_lookup(Walker.embedding_partial, self.edge_indices_right, max_norm=1)
        print(type(self.config['random_walk_length']))
        self.regularization_differences = self.left_features - self.right_features + np.random.uniform(
            -float(self.config['regularization_noise']), float(self.config['regularization_noise']),
            (self.config['random_walk_length'] - 1, self.config['dimensions']))
        self.regularization_distances = tf.norm(self.regularization_differences, ord=2, axis=1)
        self.regularization_distances = tf.reshape(self.regularization_distances, [-1, 1])
        self.regularization_loss = tf.reduce_mean(tf.matmul(tf.transpose(self.overlap), self.regularization_distances))
        return self.config['lambd'] * self.regularization_loss
