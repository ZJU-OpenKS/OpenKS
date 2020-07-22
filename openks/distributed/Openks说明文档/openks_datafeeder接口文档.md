### Class OpenKS_DataFeeder(object):

OpenKS_DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到Executor和ParallelExecutor中。reader通常返回一个minibatch大小的条目列表。在列表中每一条目都是一个样本（sample）,它是由具有一至多个特征的列表或元组组成的。

**参数：**

1. feed_list (list)：向模型输入的变量表或者变量表名

2. place (Place)：place表明是向GPU还是CPU中输入数据。

3. program (Program)：需要向其中输入数据的Program。如果为None, 会默认使用 default_main_program()。缺省值为None。

**返回：**

自身实例

**返回类型：**

OpenKS_DataFeeder()

**代码示例**：

```python
import numpy as np
import paddle
import paddle.fluid as fluid
from .datafeeder import OpenKS_DataFeeder

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

feeder = OpenKS_DataFeeder([data_1, data_2], place)

exe = fluid.Executor(place)

exe.run(startup_program)

feed_data = feeder.feed(reader())
```