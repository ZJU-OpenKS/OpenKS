# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import six

from paddle.fluid import core
from paddle.fluid.framework import Variable, default_main_program, _current_expected_place
from paddle.fluid.framework import _cpu_num, _cuda_ids
from paddle.fluid.data_feeder import DataToLoDTensorConverter

class DataFeeder(object):
    """
    DataFeeder converts the data that returned by a reader into a data
    structure that can feed into Executor. The reader is usually a
    python generator that returns a list of mini-batch data entries.

    Parameters:
        feed_list (list): Variables or names of Variables that need
            to feed.
        place (:ref:`api_fluid_CPUPlace` | :ref:`api_fluid_CUDAPlace` ):
            place indicates the device (CPU | GPU) the data will be fed into, if
            you want to feed data into GPU, please using :code:`fluid.CUDAPlace(i)`
            (:code:`i` represents the GPU id), or if you want to feed data into CPU,
            please using :code:`fluid.CPUPlace()`.
        program (:ref:`api_fluid_Program` , optional): The Program that will
            feed data into, if program is None, it will use default_main_program().
            Default None.

    Raises:
        :code:`ValueError` - If some Variables are not in this Program.

    Example:
        ..  code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid as fluid

            place = fluid.CPUPlace()
            def reader():
                for _ in range(4):
                    yield np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32'),

            main_program = fluid.Program()
            startup_program = fluid.Program()

            with fluid.program_guard(main_program, startup_program):
                data_1 = fluid.data(name='data_1', shape=[None, 2, 2], dtype='float32')
                data_2 = fluid.data(name='data_2', shape=[None, 1, 3], dtype='float32')
                out = fluid.layers.fc(input=[data_1, data_2], size=2)
                # ...
            feeder = fluid.DataFeeder([data_1, data_2], place)

            exe = fluid.Executor(place)
            exe.run(startup_program)

            feed_data = feeder.feed(reader())

            # print feed_data to view feed results
            # print(feed_data['data_1'])
            # print(feed_data['data_2'])

            outs = exe.run(program=main_program,
                            feed=feed_data,
                            fetch_list=[out])
            print(outs)

    """

    def __init__(self, feed_list, place, program=None):
        self.feed_dtypes = []
        self.feed_names = []
        self.feed_shapes = []
        self.feed_lod_level = []
        if program is None:
            program = default_main_program()
        for each_var in feed_list:
            if isinstance(each_var, six.string_types):
                each_var = program.block(0).var(each_var)
            if not isinstance(each_var, Variable):
                raise TypeError("Feed list should contain a list of variable")
            self.feed_dtypes.append(each_var.dtype)
            self.feed_names.append(each_var.name)
            self.feed_lod_level.append(each_var.lod_level)
            self.feed_shapes.append(each_var.shape)

        self.place = place

    def feed(self, iterable):
        """
        According to :code:`feed_list` of :code:`DataFeeder` and :code:`iterable` , converts
        the input into a data structure that can feed into Executor.

        Parameters:
            iterable (generator): user defined python generator to read the raw input data

        Returns:
            :code:`dict`: a :code:`dict` that contains (variable name - converted tensor) pairs

        Example:
            ..  code-block:: python

                # In this example, reader - generator will return a list of ndarray of 3 elements
                # feed API will convert each ndarray input into a tensor
                # the return result is a dict with keys: data_1, data_2, data_3
                # result['data_1']  a LoD-Tensor with shape of  [5, 2, 1, 3]. 5 is batch size, and [2, 1, 3] is the real shape of data_1.
                # result['data_2'], result['data_3'] are similar.
                import numpy as np
                import paddle.fluid as fluid

                def reader(limit=5):
                    for i in range(1, limit + 1):
                        yield np.ones([6]).astype('float32') * i , np.ones([1]).astype('int64') * i, np.random.random([9]).astype('float32')

                data_1 = fluid.data(name='data_1', shape=[None, 2, 1, 3])
                data_2 = fluid.data(name='data_2', shape=[None, 1], dtype='int64')
                data_3 = fluid.data(name='data_3', shape=[None, 3, 3], dtype='float32')
                feeder = fluid.DataFeeder(['data_1','data_2', 'data_3'], fluid.CPUPlace())


                result = feeder.feed(reader())
                print(result['data_1'])
                print(result['data_2'])
                print(result['data_3'])

        """
        converter = []
        for lod_level, shape, dtype in six.moves.zip(
                self.feed_lod_level, self.feed_shapes, self.feed_dtypes):
            converter.append(
                DataToLoDTensorConverter(
                    place=self.place,
                    lod_level=lod_level,
                    shape=shape,
                    dtype=dtype))

        for each_sample in iterable:
            assert len(each_sample) == len(converter), (
                "The number of fields in data (%d) does not match " +
                "len(feed_list) (%d)") % (len(each_sample), len(converter))
            for each_converter, each_slot in six.moves.zip(converter,
                                                           each_sample):
                each_converter.feed(each_slot)
        ret_dict = {}
        for each_name, each_converter in six.moves.zip(self.feed_names,
                                                       converter):
            ret_dict[each_name] = each_converter.done()
        return ret_dict

    def feed_parallel(self, iterable, num_places=None):
        """
        Similar with feed function, feed_parallel is used with multiple devices (CPU|GPU).
        Here :code:`iterable` is a list of python generators. The data return by each
        generator in the list will be fed into a separate device.

        Parameters:
            iterable (list|tuple): list of user-defined python generators. The element
                number should match the :code:`num_places`.
            num_places (int, optional): the number of devices. If not provided (None),
                all available devices on the machine will be used. Default None.

        Returns:
            :code:`generator`: a :code:`generator` that generate dict which contains (variable name - converted tensor) pairs,
            the total number of dicts will be generated matches with the :code:`num_places`

        .. note::
            The number of devices - :code:`num_places` should equal to the generator (element of :code:`iterable` ) number

        Example:
            ..  code-block:: python

                import numpy as np
                import paddle.fluid as fluid

                def generate_reader(batch_size, base=0, factor=1):
                    def _reader():
                        for i in range(batch_size):
                            yield np.ones([4]) * factor + base, np.ones([4]) * factor + base + 5
                    return _reader()

                x = fluid.data(name='x', shape=[None, 2, 2])
                y = fluid.data(name='y', shape=[None, 2, 2], dtype='float32')

                z = fluid.layers.elementwise_add(x, y)

                feeder = fluid.DataFeeder(['x','y'], fluid.CPUPlace())
                place_num = 2
                places = [fluid.CPUPlace() for x in range(place_num)]
                data = []
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(fluid.default_startup_program())
                program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(places=places)

                # print sample feed_parallel r result
                # for item in list(feeder.feed_parallel([generate_reader(5, 0, 1), generate_reader(3, 10, 2)], 2)):
                #     print(item['x'])
                #     print(item['y'])

                reader_list = [generate_reader(5, 0, 1), generate_reader(3, 10, 2)]
                res = exe.run(program=program, feed=list(feeder.feed_parallel(reader_list, 2)), fetch_list=[z])
                print(res)

        """
        if isinstance(self.place, core.CUDAPlace):
            places = [
                core.CUDAPlace(i)
                for i in six.moves.xrange(
                    self._get_number_of_places_(num_places))
            ]
        else:
            places = [
                core.CPUPlace()
                for _ in six.moves.xrange(
                    self._get_number_of_places_(num_places))
            ]

        if len(iterable) != len(places):
            raise ValueError("feed_parallel takes multiple mini-batches. Each "
                             "mini-batch will be feed on each device. The "
                             "number of devices and number of mini-batches "
                             "must be same.")

        place = self.place
        for p, batch in six.moves.zip(places, iterable):
            self.place = p
            yield self.feed(batch)
        self.place = place

    def _get_number_of_places_(self, num_places):
        if num_places is not None:
            return int(num_places)
        elif isinstance(self.place, core.CUDAPlace):
            return len(_cuda_ids())
        else:
            return _cpu_num()

    def decorate_reader(self,
                        reader,
                        multi_devices,
                        num_places=None,
                        drop_last=True):
        """
        Decorate the reader (generator) to fit multiple devices. The reader generate
        multiple mini-batches. Each mini-batch will be fed into a single device.

        Parameters:
            reader(generator): a user defined python generator used to get :code:`mini-batch` of data.
                A :code:`mini-batch` can be regarded as a python generator that returns batches of input
                entities, just like the below :code:`_mini_batch` in the code example.
            multi_devices(bool): indicate whether to use multiple devices or not.
            num_places(int, optional): if :code:`multi_devices` is True, you can specify the number
                of devices(CPU|GPU) to use, if multi_devices is None, the function will use all the
                devices of the current machine. Default None.
            drop_last(bool, optional): whether to drop the last round of data if it is not enough to
                feed all devices. Default True.

        Returns:
            :code:`generator`: a new :code:`generator` which return converted dicts that can be fed into Executor

        Raises:
            :code:`ValueError`: If drop_last is False and the data cannot fit devices perfectly.

        Example:
            ..  code-block:: python

                import numpy as np
                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.compiler as compiler

                def reader():
                    def _mini_batch(batch_size):
                        for i in range(batch_size):
                            yield np.random.random([16]).astype('float32'), np.random.randint(10, size=[1])

                    for _ in range(10):
                        yield _mini_batch(np.random.randint(1, 10))

                place_num = 3
                places = [fluid.CPUPlace() for _ in range(place_num)]

                # a simple network sample
                data = fluid.data(name='data', shape=[None, 4, 4], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                hidden = fluid.layers.fc(input=data, size=10)

                feeder = fluid.DataFeeder(place=places[0], feed_list=[data, label])
                reader = feeder.decorate_reader(reader, multi_devices=True, num_places=3, drop_last=True)

                exe = fluid.Executor(places[0])
                exe.run(fluid.default_startup_program())
                compiled_prog = compiler.CompiledProgram(
                         fluid.default_main_program()).with_data_parallel(places=places)

                for i,data in enumerate(reader()):
                    # print data if you like
                    # print(i, data)
                    ret = exe.run(compiled_prog, feed=data, fetch_list=[hidden])
                    print(ret)

        """

        def __reader_creator__():
            if not multi_devices:
                for item in reader():
                    yield self.feed(item)
            else:
                num = self._get_number_of_places_(num_places)
                item = []
                for batch in reader():
                    item.append(batch)
                    if len(item) == num:
                        yield list(self.feed_parallel(item, num))
                        item = []
                if not drop_last and len(item) != 0:
                    raise ValueError(
                        "The data batch which cannot fit for devices will be "
                        "dropped is not implementation. Other strategies are "
                        "not implemented")

        return __reader_creator__
