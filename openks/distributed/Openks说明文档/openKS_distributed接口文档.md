### Class RoleMakerBase(object):

类说明：

RoleMakerBase作为一个基类，用于给当前分布式训练中的进程安排一个role

函数：

def init(self)

创建RoleMakerBase对象，对实例的worker节点、server节点列表进行初始化。

参数：无

返回：无

返回类型：无

 

def is_worker(self)

Parameter Server训练中使用，判断当前节点是否是Worker节点，是则返回True，否则返回False

参数：无

返回：是否为worker节点

返回类型：bool

 

def is_server(self)

Parameter Server训练中使用，判断当前节点是否是Server节点，是则返回True，否则返回False

参数：无

返回：是否为Server节点

返回类型：bool

 

def is_first_worker(self):

判断当前节点是否是第一个worker节点，是则返回True，否则返回False

参数：无

返回：是否为第一个worker节点

返回类型：bool

 

def worker_num(self):

返回当前总共的节点数

参数：无

返回：总共的节点数

返回类型：int

 

def worker_index(self):

返回当前worker节点id

参数：无

返回：当前节点id

返回类型：int

 

def server_index(self):

返回当前server节点id

参数：无

返回：当前server节点id

返回类型：int

 

def get_trainer_endpoints (self):

返回trainer endpoints

参数：无

返回：trainer endpoints

返回类型：self._server_endpoints 实例

 

def to_string (self):

按照特定格式返回RoleMaker实例信息

参数：无

返回：实例信息

返回类型：string

 

def all_gather(self, input):

根据input值，返回trainers和pservers之间的所有gather

参数：

input: int 或者 float

返回：trainers和pservers之间的所有gather

返回类型：values组成的list

 

def all_reduce_worker(self, input, output, mode="sum"):

trainers间所有的reduce

参数：

input: list

output:list

mode:string, 有sum, min, max三种模式

返回：无

返回类型：无

 

def barrier_worker(self):

当前trainer之间的barrier

参数: 无

返回： 无

返回类型：无

 

def barrier_all (self):

所有trainer之间的barrier

参数：无

返回：无

返回类型：无



## Class PaddleCloudRoleMaker (RoleMakerBase):

类说明：

PaddleCloudRoleMaker是一个高级封装，支持使用paddle.distributed.launch或者paddle.distributed.launch_ps启动脚本



函数：

def init(self, is_collective=False)

创建PaddleCloudRoleMaker类实例

参数：is_collective：bool类型，是否为collective模式

返回：无

返回类型：无

 

def generate_role (self)

在PaddleCloudRoleMaker类实例实例中，根据类属性创建对应的分布式role

参数：无

返回：无

返回类型：无

示例：

from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet

from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.PaddleCloudRoleMaker()

fleet.init(role)

 

### Class Open_KS_ImageNet:

类说明：

Open_KS_ImageNet为指定ImageNet文件夹中的图片生成特定的类，并且将根据传入数据随机采样多张图片数据。

 

函数：

def sample(self, num_images):

参数：

num_images：进行元学习训练时，每个类别的样本个数

返回：一些列84x84x3 的numpy矩阵

返回类型：numpy array

 

def _read_image(self, name):

参数：

name：图片的文件名

返回：图片的像素级信息

返回类型：numpy array



### Class Open_KS_Character:

类说明：

Open_KS_Character为指定文件夹中的图片生成特定的类，并且将根据传入数据随机采样多张图片数据。



函数：

def sample(self, num_images):

参数：

num_images：进行元学习训练时，每个类别的样本个数

返回：一些列84x84x3 的numpy矩阵

返回类型：numpy array

 

def _read_image(self, name):

参数：

name：图片的文件名

返回：图片的像素级信息

返回类型：numpy array

 

### Class Open_KS_read:

 

类说明：

Open_KS_read将根据输入读入训练集、测试集和交叉验证集的文件返回随机采样的图片类别以及规定的每个类别的图片个数。

 

函数：

def read_dataset(data_dir):

参数：

data_dir：存放数据集的文件及位置，下层目录中有'train', 'val', 'test'三个文件夹，分别表示训练集、交叉验证集和测试集

返回：元组形式的三个地址

返回类型：tuple

 

def _read_classes(dir_path):

参数：

dir_path：具体类别图片的文件夹位置

返回：list形式的所有图片类别

返回类型：list



### Class KSDistributedFactory:

分布式算法工厂类及其工厂方法

类的方法:

def instantiation(flag):

参数 flag : int - 选择分布式优化算法的标志位

返回 BaseDistributedAlgorithm - 分布式优化算法, 基于GPU或CPU分布式计算

 

### Class BaseDistributedAlgorithm(object):

BaseDistributedAlgorithm是分布式算法实现的基类，其中包含transpiler和pslib两种类别的BaseDistributedAlgorithm的实现。

参数：mode(Mode): BaseDistributedAlgorithm的实现模式。

返回：无

 

类的方法:

def init(self, mode):

根据输入参数mode创建BaseDistributedAlgorithm对象, 对其各项属性进行初始化

参数: mode(Mode): BaseDistributedAlgorithm的实现模式。

返回: 无

 

def is_first_worker(self):

判断当前节点是否是第一个worker节点，是则返回True，否则返回False

参数：无

返回：bool, 是否为第一个worker节点

 

def _is_worker(self)

判断当前节点是否是Worker节点，是则返回True，否则返回False

参数：无

返回：bool, 是否为worker节点

 

def worker_num(self):

返回当前总共的节点数

参数：无

返回：int, 总共的节点数

 

def worker_endpoints(self, to_string=False):

获取当前服务器端点，如["127.0.0.1:1001 "、" 127.0.0.1:1002"]

参数：to_string(string), RoleMaker实例信息

返回：string, 服务器端点

 

def is_server(self)

判断当前节点是否是Server节点，是则返回True，否则返回False

参数：无

返回：bool, 是否为Server节点

 

def split_files(self, files):

在分布式训练前对文件进行拆分。

参数：file(list): 需要读取的文件列表

返回：list, 文件所属的worker

 

def init(self, role_maker=None):

初始化用于识别当前节点角色(如worker或server)的RoleMaker, 在用户的python脚本中应当仅被调用一次。

参数: role_maker(RoleMakerBase) : RoleMakerBase的子类。

返回: 无

 

def all_reduce_worker(self, input, output):

worker之间的all reduce, 仅支持一维矩阵。

参数: 

input(list | numpy.array): 一维矩阵

output(list | numpy.array): 一维矩阵

返回: 无

 

def barrier_workers(self):

worker之间的barrier

参数: 无

返回: 无

 

 

### Class BaseDistributedOptimizer(object):

BaseDistributedOptimizer是一个paddle.fluid.optimizer的封装, 用户应该将paddle.fluid. optimizer传递给BaseDistributedOptimizer, 实现了minimize()函数。

BaseDistributedOptimizer作为优化器运行分布式训练。优化信息将存储在BaseDistributedAlgorithm()实例，它保存当前分布式训练的全局信息。

参数:

optimizer (Optimizer): optimizer的子类。

strategy (any): 用户为优化器定义的策略。

返回:无

 

类的方法:

def __init__(self, optimizer, strategy=None):

根据输入的optimizer和strategy初始化BaseDistributedOptimizer.

 

 

 

 

### Class CPUDistributedAlgorithm(BaseDistributedAlgorithm):

和fluid.transpiler.DistributeTranspiler.相兼容的BaseDistributedAlgorithm子类.

类的方法:

def init(self):

创建BaseDistributedAlgorithm对象, 对其各项属性进行初始化。

参数：无

返回：无

 

def init_worker(self):

init_worker是在训练前运行的一系列步骤。首先，等待所有pserver完全启动；第二，运行executor初始化startup program；第三，等待所有worker初始化。

参数：无

返回：无

 

def init_server(self, model_dir=None):

init_server是启动pserver之前的一系列步骤。首先，运行executor初始化startup program；第二，如果model_dir不为空，它将从中加载参数用于增量训练。

参数：model_dir(str)：模型的路径

返回：无

 

def run_server(self):

run_server操作executor初始化pserver的主程序。

参数：无

返回：无

 

def stop_worker(self):

关闭当前executor，对于分布式训练，这一方法将释放与当前trainer相关的pserver的资源。

参数：无

返回：无

 

def distributed_optimizer(self, optimizer, strategy=None):

分布式训练的优化器。对于分布式训练，这种方法将建立一个新的BaseDistributedOptimizer实例。它具有基础的优化器功能和分布式训练的特殊功能。

参数:

optimizer(Optimizer)：为初始化服务器运行的executor。

strategy(DistributeTranspilerConfig)：分布式优化器的策略。

返回：GeneralDistributedOptimizer：BaseDistributedOptimizer的子类。

 

def save_inference_model(self, executor, dirname, feeded_var_names, target_vars, main_program = None, export_for_deployment = True)

修剪给定的main_program以构建一个专用于推理的新程序，然后通过executor将它和所有相关参数保存到给定的dirname中。

参数：

executor(Executor)：当前程序的executor

dirname(str)：保存到的路径dirname

feeded_var_names(str)：被加入的变量名

target_vars(Variable)：目标变量

main_program(Program)：当前主程序

export_for_deployment(bool)：用于部署的导出

返回：无

 

def save_persistables(self，executor，dirname，main_program=None):

该函数从给定的main_program中选择所有presistable = True的变量并储存到文件夹dir_name或文件filename中。dir_name用于指定保存持久变量的文件夹。如果要将变量保存在单独的文件中，请将“filename”设置为“None”；如果想将所有变量保存在一个文件中，请使用“filename”

参数：

executor(Executor)：当前程序的executor

dirname(str)：保存变量的路径名

main_program(Program)：给定的主程序

返回：无

 

### Class GeneralDistributedOptimizer(BaseDistributedOptimizer):

GeneralDistributedOptimizer是BaseDistributedOptimizer类的子类

类的方法：

backward(self, loss, startup_program=None, parameter_list=None, no_grad_set=None, callbacks =None):

minimize的第一部分,进行auto-diff来为当前的program附加反向传播的运算符

参数:

loss (Variable): 用于进行优化的损失

startup_program (Program): 用于对“parameter_list”中的参数进行优化的启动程序

parameter_list (list): 需要更新的参数列表

no_grad_set (set | None): 应忽略的变量集合

callbacks (list | None): 为一个参数追加反向传播运算符时要运行的可调用列表

返回:

list: (param, grad)，对的列表, 其中grad是反向传播的输出。

 

apply_gradients(self, params_grads): 

“minimize”的第二部分, 为给定的“param_grads”对添加优化运算符。

参数:

params_grads (list):要进行优化的(param，grad)对列表.

返回:

list:附加到当前程序的操作符列表.

 

minimize(self, loss, startup_program=None, parameter_list=None, no_grad_set = None):

通过更新parameter_list来添加操作以最小化loss, 该方法把该方法将接口“backward()”和“apply_gradients()”合二为一.

参数:

loss (Variable):运行优化的损失变量.

startup_program (Program):用于初始化在“parameter_list”中的参数的启动程序.

parameter_list (list):要更新的变量列表.

no_grad_set (set | None):应忽略的变量集合.

返回:

tuple:附加的运算符列表(optimize_ops，params_grads)和用于优化的变量对列表(param，grad).

 

### Class GPUDistributedAlgorithm(BaseDistributedAlgorithm)：

继承BaseDistributedAlgorithm的, 用于GPU分布式训练的类, 主要包括创建分布式优化器, 执行分布式算法等的方法:



类的方法:

def init(self):

创建GPUDistributedAlgorithm对象, 对其各项属性进行初始化。

参数：无

返回：无

 

def distributed_optimizer(self, optimizer, strategy=None):

封装local optimizer, 返回分布式优化器BaseDistributedOptimizer, 提供分布式的优化支持

参数: 

optimizer: Optimizer - 本地优化器

strategy: BuildStrategy - 分布式优化的配置策略

返回: BaseDistributedOptimizer - 分布式优化器

 

def save_inference_model(self, executor, dirname, feeded_var_names, target_vars, main_program = None, export_for_deployment = True):

修剪给定的main_program以构建一个专用于推理的新程序，然后通过executor将它和所有相关参数保存到给定的dirname中。

参数：

executor : Executor - 当前程序的executor

dirname : str - 保存到的路径dirname

feeded_var_names : str - 被加入的变量名

target_vars : Variable - 目标变量

main_program : Program - 当前主程序

export_for_deployment : bool - 用于部署的导出

返回：无

 

def save_persistables(self，executor，dirname，main_program=None):

该函数从给定的main_program中选择所有presistable = True的变量并储存到文件夹dir_name或文件filename中。dir_name用于指定保存持久变量的文件夹。如果要将变量保存在单独的文件中，请将“filename”设置为“None”；如果想将所有变量保存在一个文件中，请使用“filename”

参数：

executor : Executor - 当前程序的executor

dirname : str - 保存变量的路径名

main_program : Program - 给定的主程序

返回：无

 

 

 

#### Class HeterogeneousDistributedOptimizer(BaseDistributedOptimizer):

HeterogeneousDistributedOptimizer是BaseDistributedOptimizer类的子类, 用于GPU异构计算的分布式优化器

类的方法:

def backward(self, loss, startup_program=None, parameter_list=None, no_grad_set=None, callbacks =None):

minimize的第一部分,进行auto-diff来为当前的program附加反向传播的运算符

参数:

loss : Variable - 用于进行优化的损失

startup_program : Program - 用于对“parameter_list”中的参数进行优化的启动程序

parameter_list : list - 需要更新的参数列表

no_grad_set : set | None - 应忽略的变量集合

callbacks : list | None - 为一个参数追加反向传播运算符时要运行的可调用列表

返回: list: tuple(param, grad) - 对的列表, 其中grad是反向传播的输出。

 

def apply_gradients(self, params_grads): 

“minimize”的第二部分, 为给定的“param_grads”对添加优化运算符。

参数: params_grads : list - 要进行优化的(param，grad)对列表.

返回: list - 附加到当前程序的操作符列表.



def minimize(self, loss, startup_program=None, parameter_list=None, no_grad_set = None):

通过更新parameter_list来添加操作以最小化loss, 该方法将接口“backward()”和

“apply_gradients()”合二为一.

参数:

loss : Variable - 运行优化的损失变量.

startup_program : Program - 用于初始化在“parameter_list”中的参数的启动程序

parameter_list : list - 要更新的变量列表.

no_grad_set : (set | None) - 应忽略的变量集合.

返回: tuple - 附加的运算符列表(optimize_ops，params_grads)和用于优化的变量对列表(param，grad).



def _try_to_compile(self, startup_program, main_program):

内部方法，根据用户配置生成编译配置，调用_transpile方法将startup_program 和 

main_program转换为分布式程序，调用编译后程序的 CompiledProgram.with_data_parallel方法构建数据并行。

参数:

startup_program : Program - 用于初始化在“parameter_list”中的参数的启动程序

main_program: Program - 需要被程序的主要部分, 单机Program的算子集合

返回: 无



def _transpile(self, startup_program, main_program)

该方法调用 DistributeTranspiler 将程序转换为分布式程序，向程序中添加分布式执行相关的变量。

参数:

startup_program : Program - 需要被转换的程序的初始化部分

main_program : Program - 需要被程序的主要部分, 单机Program的算子集合

返回: 无

 