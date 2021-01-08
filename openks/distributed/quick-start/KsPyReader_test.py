# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import paddle
import paddle.fluid as fluid
import numpy as np
from KsPyReader import OpenKS_PyReader

EPOCH_NUM = 3
ITER_NUM = 5
BATCH_SIZE = 3

def network(image, label):
    # 用户定义网络，此处以softmax回归为例
    predict = fluid.layers.fc(input=image, size=10, act='softmax')
    return fluid.layers.cross_entropy(input=predict, label=label)

def reader_creator_random_image_and_label(height, width):
    def reader():
        for i in range(ITER_NUM):
            fake_image = np.random.uniform(low=0,
                                           high=255,
                                           size=[height, width])
            fake_label = np.ones([1])
            yield fake_image, fake_label
    return reader

image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

reader = OpenKS_PyReader(feed_list=[image, label],
                           capacity=4,
                           iterable=False)

user_defined_reader = reader_creator_random_image_and_label(784, 784)
reader.decorate_sample_list_generator(
    paddle.batch(user_defined_reader, batch_size=BATCH_SIZE))

loss = network(image, label)
executor = fluid.Executor(fluid.CPUPlace())
executor.run(fluid.default_startup_program())
for i in range(EPOCH_NUM):
    reader.start()
    while True:
        try:
            executor.run(feed=None)
        except fluid.core.EOFException:
            reader.reset()
            break