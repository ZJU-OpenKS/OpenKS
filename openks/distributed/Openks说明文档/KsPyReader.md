### Class OpenKS_PyReader:
说明： OpenKS_PyReader 对象用于为计算图(Program)输入数据。根据iterable设置为True或者False，可以指定该对象是作为计算图中的一个Operator或是独立于计算图。创建对象后，采用装饰器设置其数据源。可以采用`decorate_sample_generator(sample_generator, batch_size, drop_last=True, places=None)`,  `decorate_sample_list_generator(reader, places=None)`, `decorate_batch_generator(reader, places=None)` 三种不同的装饰器。

#### 初始化
```
class OpenKS_PyReader(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False)
```

参数：

* feed_list (list(Variable)|tuple(Variable)): 需要feed的变量列表，是由`fluid.layers.data()`创建的占位符。
* capacity (int): OpenKS_PyReader 对象内部维护队列的容量大小。单位是batch数量。若reader读取速度较快，建议设置较大的capacity值。
* use_double_buffer (bool): 是否使用 double_buffer_reader 。设置为True时， OpenKS_PyReader 会异步地预读取下一个batch的数据，可加速数据读取过程，但同时会占用少量的CPU/GPU存储，即一个batch输入数据的存储空间。
* iterable (bool): 所创建的DataLoader对象是否可迭代。iterable=False时，该 OpenKS_PyReader 对象作为计算图中的一个Operator，见代码示例1。 当iterable=True时，该 OpenKS_PyReader 对象独立于计算图，是一个可迭代的python生成器，见代码示例2。
* return_list (bool): 每个设备上的数据是否以list形式返回。仅在iterable = True模式下有效。若return_list = False，每个设备上的返回数据均是str -> LoDTensor的映射表，其中映射表的key是每个输入变量的名称。若return_list = True，则每个设备上的返回数据均是list(LoDTensor)。推荐在静态图模式下使用return_list = False，在动态图模式下使用return_list = True。

返回: 被创建的reader对象

返回类型： reader (Reader)

代码示例：

1. iterable=False，创建的 OpenKS_PyReader 对象作为一个Operator将被插入到计算图(program)中。在训练时，用户应该在每个epoch之前调用 `start()` ，并在epoch结束时捕获 `Executor.run()` 抛出的 `fluid.core.EOFException` 。一旦捕获到异常，用户应该调用 `reset()` 手动重置reader。

```
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
```

2. iterable=True，创建的对象与计算图(Program)分离。Program中不会插入任何算子。在本例中，创建的reader是一个可迭代的python生成器。直接用for循环将每次迭代的数据feed进Executor中，`Executor.run(feed=...)`。
```
import paddle
import paddle.fluid as fluid
import numpy as np
from KsPyReader import OpenKS_PyReader

EPOCH_NUM = 3
ITER_NUM = 5
BATCH_SIZE = 10

def network(image, label):
     # 用户定义网络，此处以softmax回归为例
     predict = fluid.layers.fc(input=image, size=10, act='softmax')
     return fluid.layers.cross_entropy(input=predict, label=label)

def reader_creator_random_image(height, width):
    def reader():
        for i in range(ITER_NUM):
            fake_image = np.random.uniform(low=0, high=255, size=[height, width]),
            fake_label = np.ones([1])
            yield fake_image, fake_label
    return reader

image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
reader = OpenKS_PyReader(feed_list=[image, label], capacity=4, iterable=True, return_list=False)

user_defined_reader = reader_creator_random_image(784, 784)
reader.decorate_sample_list_generator(
    paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
    fluid.core.CPUPlace())
loss = network(image, label)
executor = fluid.Executor(fluid.CPUPlace())
executor.run(fluid.default_startup_program())

for _ in range(EPOCH_NUM):
    for data in reader():
        executor.run(feed=data, fetch_list=[loss])
```

#### 装饰器

```
decorate_sample_generator(sample_generator, batch_size, drop_last=True, places=None)
```
参数:
* sample_generator (generator) – Python生成器，yield 类型为 tuple(numpy.ndarray)
* batch_size (int) – batch size，必须大于0
* drop_last (bool) – 如果设置为True，当最后一个batch中的样本数量小于batch size时，丢弃最后一个batch
* places (None|list(CUDAPlace)|list(CPUPlace)) – 位置列表。当 iterable=True 时必须被提供

代码示例：
```
def random_image_and_label_generator(height, width):
    def generator():
        for i in range(ITER_NUM):
            fake_image = np.random.uniform(low=0,
                                           high=255,
                                           size=[height, width])
            fake_label = np.array([1])
            yield fake_image, fake_label
    return generator

image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
reader = OpenKS_PyReader(feed_list=[image, label], capacity=4, iterable=True)
user_defined_generator = random_image_and_label_generator(784, 784)
reader.decorate_sample_generator(user_defined_generator,
                                 batch_size=BATCH_SIZE,
                                 places=[fluid.CPUPlace()])
```

```
decorate_sample_list_generator(reader, places=None)
```
参数：
* reader (generator) – Python生成器，yield 类型为 list[tuple(numpy.ndarray)]， list的长度为batch size。可以采用`paddle.batch(reader, batch_size, drop_last=False)` 将yield 类型为 tuple(numpy.ndarray)的reader转换为此处需要的reader
* places (None|list(CUDAPlace)|list(CPUPlace)) – 位置列表。当 iterable=True 时必须被提供

代码示例：
```
def random_image_and_label_generator(height, width):
    def generator():
        for i in range(ITER_NUM):
            fake_image = np.random.uniform(low=0,
                                           high=255,
                                           size=[height, width])
            fake_label = np.ones([1])
            yield fake_image, fake_label
    return generator

image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
reader = OpenKS_PyReader(feed_list=[image, label], capacity=4, iterable=True)

user_defined_generator = random_image_and_label_generator(784, 784)
reader.decorate_sample_list_generator(
    paddle.batch(user_defined_generator, batch_size=BATCH_SIZE),
    fluid.core.CPUPlace())
```

```
decorate_batch_generator(reader, places=None)
```
参数
* reader (generator) – Python生成器， yield 类型为 tuple(numpy.ndarray) 或 tuple(LoDTensor), 其中 numpy.ndarray 或LoDTensor的shape应包含batch size 这一维度。
* places (None|list(CUDAPlace)|list(CPUPlace)) – 位置列表。当 iterable=True 时必须被提供

代码示例：
```
def random_image_and_label_generator(height, width):
    def generator():
        for i in range(ITER_NUM):
            batch_image = np.random.uniform(low=0,
                                            high=255,
                                            size=[BATCH_SIZE, height, width])
            batch_label = np.ones([BATCH_SIZE, 1])
            batch_image = batch_image.astype('float32')
            batch_label = batch_label.astype('int64')
            yield batch_image, batch_label
    return generator

image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
reader = OpenKS_PyReader(feed_list=[image, label], capacity=4, iterable=True)

user_defined_generator = random_image_and_label_generator(784, 784)
reader.decorate_batch_generator(user_defined_generator, fluid.CPUPlace())
```
