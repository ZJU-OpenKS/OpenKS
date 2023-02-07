import tensorflow as tf
import numpy as np

def glorot_init(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float64)
    return tf.Variable(initial, name=name)

def zeros_init(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float64)
    return tf.Variable(initial, name)

