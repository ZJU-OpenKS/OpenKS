# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import paddle.fluid as fluid

__all__ = ["KsPyReader"]

class DataLoaderBase(object):
    def __init__(self):
        self._places = None

    def __call__(self):
        return self

    def next(self):
        '''
        Get the next item in the DataLoader object. This method    
        should not be called by users directly. It is used for
        implementing iterator protocol of Python 2.x inside
        PaddlePaddle framework.
        '''
        return self.__next__()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class KsPyReader(DataLoaderBase):
    """
    Create a reader object for data feeding in Python. 
    Data would be prefetched using Python thread and be pushed
    into a queue asynchronously. 

    Args:  
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`fluid.layers.data()`.
        capacity (int): capacity of the queue maintained in PyReader.
            The unit is batch number. Set larger capacity if your reader 
            is fast. 
        use_double_buffer (bool): whether to use double_buffer_reader, which
            would speed up data feeding but occupies a little more CPU or 
            GPU memory.
        iterable (bool): whether the created PyReader is iterable. 
            If iterable = False, Operators would be inserted into the program. 
            User should call `start()` before each epoch and catch 
            `fluid.core.EOFException` thrown by `Executor.run()` when epoch ends. 

            If iterable=True, the created PyReader object is decoupled with the 
            program. No operator would be inserted into the program. In this case, 
            the created reader is a Python generator, which is iterable. User 
            should feed the data yielded from PyReader object into Executor.run(feed=...)
            
        return_list (bool): whether the return value on each device is 
            presented as a list. It is only valid when iterable=True. 

    Returns:
        the created reader object.
    """
    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False):
        self._loader = fluid.io.DataLoader.from_generator(
            feed_list, capacity, use_double_buffer, iterable, return_list)
    
    @property
    def queue(self):
        return self._loader.queue

    @property
    def iterable(self):
        return self._loader.iterable

    def __iter__(self):
        return self._loader.__iter__()

    def __next__(self):
        return self._loader.__next__()
    
    def start(self):

        self._loader.start()

    def reset(self):

        self._loader.reset()

    def decorate_sample_generator(self,
                                  sample_generator,
                                  batch_size,
                                  drop_last=True,
                                  places=None):

        self._loader.set_sample_generator(sample_generator, batch_size,
                                          drop_last, places)

    def decorate_sample_list_generator(self, reader, places=None):

        self._loader.set_sample_list_generator(reader, places)

    def decorate_batch_generator(self, reader, places=None):

        self._loader.set_batch_generator(reader, places)

