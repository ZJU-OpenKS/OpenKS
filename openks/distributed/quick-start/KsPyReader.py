# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import paddle.fluid as fluid


class OpenKS_PyReader(object):
    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False):
        self._pyreader = fluid.io.PyReader(feed_list,capacity,use_double_buffer,iterable,return_list)
    
    @property
    def queue(self):
        return self._pyreader.queue

    @property
    def iterable(self):
        return self._pyreader.iterable

    def __iter__(self):
        return self._pyreader.__iter__()

    def __next__(self):
        return self._pyreader.__next__()
    
    def start(self):

        self._pyreader.start()

    def reset(self):

        self._pyreader.reset()

    def decorate_sample_generator(self,
                                  sample_generator,
                                  batch_size,
                                  drop_last=True,
                                  places=None):

        self._pyreader.decorate_sample_generator(sample_generator, batch_size,
                                          drop_last, places)

    def decorate_sample_list_generator(self, reader, places=None):

        self._pyreader.decorate_sample_list_generator(reader, places)

    def decorate_batch_generator(self, reader, places=None):

        self._pyreader.decorate_batch_generator(reader, places)

