import paddle.fluid as fluid

class KSDataFeeder(fluid.DataFeeder):
    def __init__(self, feed_list, place, program=None):
        super(KSDataFeeder, self).__init__(feed_list, place, program)