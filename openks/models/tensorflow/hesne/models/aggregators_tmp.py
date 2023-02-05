import tensorflow as tf

class MeanAggregator(object):
    def __init__(self, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

    def __call__(self, inputs):
        self_vecs, neigh_vecs = inputs
        #self_vecs shape: [size, h_dim]
        #neigh_vecs shape: [num_neigh, h_dim]
        self_vecs_num = tf.shape(self_vecs)[0]
        neigh_means = tf.reduce_mean(neigh_vecs, axis=0, keepdims=True)
        # self_vecs_num = tf.expand_dims(self_vecs_num, -1)
        output = tf.tile(neigh_means, [self_vecs_num, 1])
        return output

class AttentionAggregator(object):
    def __init__(self, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

    def __call__(self, inputs):
        #self_vecs shape: [size, h_dim]
        #neigh_vecs shape: [num_neigh, h_dim]
        self_vecs, neigh_vecs = inputs

        query = tf.expand_dims(self_vecs, 1)
        score = tf.matmul(query, neigh_vecs, transpose_b=True)
        score = tf.nn.softmax(score, dim=-1)

        output = tf.matmul(score, neigh_vecs)
        output = tf.squeeze(output, [1])
        return output
