# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import paddle.fluid as fluid

class OpenKS_DataFeeder(object):
    def __init__(self, feed_list, place, program=None):
        self.df = fluid.DataFeeder(feed_list, place, program)

    def feed(self, iterable):

        return self.df.feed(iterable)

    def feed_parallel(self, iterable, num_places=None):

        self.df.feed_parallel(iterable, num_places)

    def _get_number_of_places_(self, num_places):

        return self.df._get_number_of_places_(num_places)

    def decorate_reader(self,
                        reader,
                        multi_devices,
                        num_places=None,
                        drop_last=True):
        return self.df.decorate_reader(reader,
                        multi_devices,
                        num_places,
                        drop_last)