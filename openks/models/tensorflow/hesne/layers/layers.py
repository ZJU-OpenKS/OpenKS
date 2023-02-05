import tensorflow as tf
from utils.inits import zeros_init
import numpy as np

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):

    def __init__(self, input_dim, output_dim, keep=1., weight_decay=0.,
                 act=None, placeholders=None, bias=False, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.keep = keep
        self.weight_decay = weight_decay

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float64,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros_init([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.nn.dropout(x, self.keep)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
        if self.act:
            output = self.act(output)

        return output

# class TimeDymLayer(Layer):
#     def __init__(self, emb_dim, keep=1., weight_decay=0., act=tf.nn.sigmoid, bias=True, **kwargs):
#         super(TimeDymLayer, self).__init__(**kwargs)
#         self.keep = keep
#         self.weight_decay = weight_decay
#         self.act = act
#         self.bias= bias
#         self.emb_dim = emb_dim
#         # self.boundary = -0.00001
#         with tf.variable_scope(self.name + '_vars'):
#             self.vars['weights_t'] = tf.get_variable('weights_t', shape=(1, emb_dim),
#                                                      dtype=tf.float64,
#                                                      initializer=tf.contrib.layers.xavier_initializer(),
#                                                      # initializer=tf.random_uniform_initializer(0,1),
#                                                      # constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
#                                                      regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#             self.vars['weights_emb'] = tf.get_variable('weights_emb', shape=(emb_dim*4, emb_dim),
#                                                        dtype=tf.float64,
#                                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                                        regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#             if self.bias:
#                 self.vars['bias'] = zeros_init([emb_dim], name='bias')
#
#
#     def _call(self, inputs):
#         x_deltat, x_static, x_dynamic = inputs[0], inputs[1], inputs[2]
#         x_delta_emb = x_dynamic - x_static
#         x_dot_emb = tf.multiply(x_static, x_dynamic)
#         x_feat_emb = tf.concat([x_static, x_dynamic, x_delta_emb, x_dot_emb], 1)
#         x_feat_diff = tf.matmul(x_feat_emb, self.vars['weights_emb'])
#         # self.vars['weights_t'] = tf.cond( , self.vars['weights_t'], self.boundary)
#         x_feat_t = self.act(tf.matmul(x_deltat, self.vars['weights_t']))
#         x_feat = x_feat_diff+x_feat_t
#         # x_feat = x_feat_t
#         if self.bias:
#             x_feat += self.vars['bias']
#         w_gate = self.act(x_feat)
#         return tf.add(tf.multiply(w_gate, x_static), tf.multiply(1-w_gate, x_dynamic))
#
# class EventLayer(Layer):
#     def __init__(self, emb_dim, keep=1., weight_decay=0., act=tf.nn.sigmoid, bias=True, **kwargs):
#         super(EventLayer, self).__init__(**kwargs)
#         self.keep = keep
#         self.weight_decay = weight_decay
#         self.act = act
#         self.bias= bias
#         # self.emb_dim = emb_dim
#         # self.boundary = -0.00001
#         with tf.variable_scope(self.name + '_vars'):
#             # self.vars['weights_event'] = tf.get_variable('weights_event', shape=(emb_dim, emb_dim),
#             #                                          dtype=tf.float64,
#             #                                          initializer=tf.contrib.layers.xavier_initializer(),
#             #                                          # initializer=tf.random_uniform_initializer(-1,1),
#             #                                          # constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
#             #                                          regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#
#             self.vars['weights_event'] = tf.get_variable('weights_event', shape=(emb_dim, emb_dim),
#                                                          dtype=tf.float64,
#                                                          initializer=tf.contrib.layers.xavier_initializer(),
#                                                          # initializer=tf.random_uniform_initializer(-1,1),
#                                                          # constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
#                                                          regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#
#             # self.vars['weights_input'] = tf.get_variable('weights_input', shape=(emb_dim, emb_dim),
#             #                                           dtype=tf.float64,
#             #                                           initializer=tf.contrib.layers.xavier_initializer(),
#             #                                           # initializer=tf.random_uniform_initializer(-1, 1),
#             #                                           regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#
#             self.vars['weights_emb'] = tf.get_variable('weights_emb', shape=(emb_dim, emb_dim),
#                                                        dtype=tf.float64,
#                                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                                        # initializer=tf.random_uniform_initializer(-1,1),
#                                                        regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#
#             # self.vars['weights_type'] = tf.get_variable('weights_type', shape=(type_dim, emb_dim),
#             #                                             dtype=tf.float64,
#             #                                             # initializer=tf.contrib.layers.xavier_initializer(),
#             #                                             initializer=tf.random_uniform_initializer(-1,1),
#             #                                             regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#
#             # self.vars['weights_add'] = tf.get_variable('weights_add', shape=(emb_dim, emb_dim),
#             #                                             dtype=tf.float64,
#             #                                             initializer=tf.contrib.layers.xavier_initializer(),
#             #                                             # initializer=tf.random_uniform_initializer(-1,1),
#             #                                             regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
#             if self.bias:
#                 self.vars['bias'] = zeros_init([emb_dim], name='bias')
#
#
#     def _call(self, inputs):
#         # x_event, x_input, x_emb, x_cel= inputs[0], inputs[1], inputs[2], inputs[3]
#
#         x_event, x_emb, x_cel= inputs[0], inputs[1], inputs[2]
#         x_feat = tf.matmul(x_event, self.vars['weights_event']) + tf.matmul(x_emb, self.vars['weights_emb'])
#
#         # x_delta_emb = x_dynamic - x_static
#         # x_dot_emb = tf.multiply(x_static, x_dynamic)
#         # x_feat_emb = tf.concat([x_static, x_dynamic], 1)
#         # x_feat_diff = tf.matmul(x_feat_emb, self.vars['weights_emb'])
#         # self.vars['weights_t'] = tf.cond( , self.vars['weights_t'], self.boundary)
#         # x_feat_t = self.act(tf.matmul(x_deltat, self.vars['weights_t']))
#         # x_feat = x_feat_diff+x_feat_t
#         # x_feat = x_feat_t
#         # x_feat = tf.matmul(x_input, self.vars['weights_input']) + tf.matmul(x_emb, self.vars['weights_emb'])
#
#         # x_feat = tf.matmul(x_event, self.vars['weights_event'])
#         if self.bias:
#             x_feat += self.vars['bias']
#         w_gate = self.act(x_feat)
#         # x_cel_new = tf.matmul(tf.add(x_cel, tf.multiply(w_gate, x_event)), self.vars['weights_add'])
#         x_cel_new = tf.add(x_cel, tf.multiply(w_gate, x_event))
#         # if self.bias:
#         #     x_cel_new += self.vars['bias']
#         return x_cel_new, tf.tanh(x_cel_new)
#         # return tf.add(tf.multiply(1-w_gate,x_cel), tf.multiply(w_gate, x_input))
#         # return tf.add(tf.multiply(w_gate, x_cel), tf.multiply(1-w_gate, x_deltaemb))
#         # return tf.add(tf.multiply(w_gate, sent_static), tf.multiply(1-w_gate, sent_dynamic))
#
# # class Triangularize(Layer):
# #     def __init__(self, size, **kwargs):
# #         self.size = size
# #         self.filter = np.ones((size, size), dtype=np.float32)
# #         self.filter = np.triu(self.filter, 1)  # upper triangle with zero diagonal
# #         self.filter = self.filter.reshape((1, size, size))
# #         super(Triangularize, self).__init__(**kwargs)
# #
# #     def build(self, input_shape):
# #         # input_shape: (None, size, size)
# #         pass
# #
# #     def _call(self, x):
# #         return x * self.filter
# #
# #     def get_output_shape_for(self, input_shape):
# #         # input_shape: (None, size, size)
# #         return input_shape


class Triangularize(Layer):
    def __init__(self, **kwargs):
        super(Triangularize, self).__init__(**kwargs)

    def _call(self, x):
        # size_x = tf.shape(x)[1]
        # size_x = tf.stack([size_x, size_x])
        # filter_x = tf.ones(shape=size_x, dtype=tf.float64)
        filter_x = tf.ones_like(x)
        filter_a = tf.matrix_band_part(filter_x, 0, -1)
        filter_b = tf.matrix_band_part(filter_x, 0, 0)
        filter_x = filter_a - filter_b
        return x * filter_x

class Reservior_predict(Layer):
    def __init__(self, output_dim, **kwargs):
        super(Reservior_predict, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.vars['reservior_dense_layer1'] = tf.layers.Dense(units=self.output_dim, activation=None, use_bias=False, name='reservior_dense_self')
        self.vars['reservior_dense_layer2'] = tf.layers.Dense(units=self.output_dim, activation=None, use_bias=False, name='reservior_dense_neigh')
        self.vars['reservior_dense_layer3'] = tf.layers.Dense(units=1, activation=None, use_bias=True, name='reservior_dense_score')

    def _call(self, inputs):
        node_states, sub_event_states = inputs
        node_states = tf.expand_dims(self.vars['reservior_dense_layer1'](node_states), 1)
        sub_event_states = tf.expand_dims(self.vars['reservior_dense_layer2'](sub_event_states), 0)
        reservior_score = tf.squeeze(self.vars['reservior_dense_layer3'](node_states+sub_event_states), [2])
        return reservior_score

class History_attention(Layer):
    def __init__(self, output_dim, **kwargs):
        super(History_attention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.vars['hisattention_dense_layer'] = tf.layers.Dense(units=self.output_dim, activation=tf.sigmoid, use_bias=True, name='hisattention_dense')
        # self.vars['hisattention_dense_layer2'] = tf.layers.Dense(units=self.output_dim, activation=None, use_bias=False, name='hisattention_dense_his')

    def _call(self, inputs):
        node_hid, node_his = inputs
        node_concat = tf.concat([node_hid, node_his], 1)
        output_a = self.vars['hisattention_dense_layer'](node_concat)
        return output_a







