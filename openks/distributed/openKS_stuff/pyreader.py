import paddle.fluid as fluid

# pyreader=fluid.io.PyReader
class KSPyReader(fluid.reader.PyReader):
    def __init__(self, feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False):
        super(KSPyReader, self).__init__(feed_list, capacity, use_double_buffer, iterable, return_list)