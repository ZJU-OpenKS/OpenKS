from layers.layers import Layer
from utils.inits import *
import tensorflow as tf

class MeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, keep=1., name=None, weight_decay=0., act=tf.nn.relu, bias=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)
        if name is not None:
            self.name = '/'+name
        else:
            self.name = ''
        self.keep = keep
        self.bias = bias
        self.act = act
        self.weight_decay = weight_decay
        with tf.variable_scope(self.name + '_vars'):
            self.vars['neigh_weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                                   dtype=tf.float64,
                                                   initializer=tf.contrib.layers.xavier_initializer(),
                                                   regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            if self.bias:
                self.vars['neigh_bias'] = zeros_init([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        self_vecs, neigh_vecs = inputs
        #self_vecs shape: [batch_size, h_dim]
        #neigh_vecs shape: [batch_size, max_num_neigh, h_dim]
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        neigh_means = tf.nn.dropout(neigh_means, self.keep)
        # output = neigh_means
        output = tf.concat([self_vecs, neigh_means], axis=1)
        output = tf.matmul(output, self.vars['neigh_weights'])
        if self.bias:
            output += self.vars['neigh_bias']
        return output

class AttentionAggregator(Layer):
    def __init__(self, output_dim, keep=1., name=None, weight_decay=0., act=tf.nn.relu, bias=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)
        if name is not None:
            self.name = '/'+name
        else:
            self.name = ''
        self.output_dim = output_dim
        self.keep = keep
        self.bias = bias
        self.act = act
        self.weight_decay = weight_decay
        with tf.variable_scope(self.name + '_vars'):
            self.vars['dense_layer1'] = tf.layers.Dense(units=self.output_dim, activation=None, use_bias=False, name='dense_self')
            self.vars['dense_layer2'] = tf.layers.Dense(units=self.output_dim, activation=None, use_bias=False, name='dense_neigh')
            self.vars['dense_layer3'] = tf.layers.Dense(units=self.output_dim, activation=tf.nn.sigmoid, use_bias=False, name='dense_neigh_deltatime')
            self.vars['dense_layer4'] = tf.layers.Dense(units=1, activation=tf.nn.relu, use_bias=True, name='dense_score')
        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        self_vecs, neigh_vecs, neigh_deltatime = inputs
        self_vecs_query = self.vars['dense_layer1'](tf.expand_dims(self_vecs,1))
        neigh_vecs_memory = self.vars['dense_layer2'](neigh_vecs)
        neigh_time_memory = self.vars['dense_layer3'](tf.expand_dims(neigh_deltatime, 2))
        score = self.vars['dense_layer4'](self_vecs_query + neigh_vecs_memory + neigh_time_memory)
        # score = self.vars['dense_layer4'](self_vecs_query + neigh_vecs_memory)
        score = tf.nn.softmax(score, axis=1)
        # score_e = tf.exp(score)*tf.expand_dims(neigh_mask, 2)
        # score_sum = tf.reduce_sum(score_e, axis=1, keepdims=True)
        # score = score_e/score_sum
        neigh_attentions = tf.squeeze(tf.matmul(score, neigh_vecs, transpose_a=True), [1])
        output = neigh_attentions
        # min_index = tf.squeeze(tf.argmin(score, axis=1))
        score = tf.squeeze(score)
        return output, score
