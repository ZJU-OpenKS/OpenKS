# 抽象类

_StrategyBase:

说明: 该类是抽象类。所有的CPU/GPU策略都会继承这个抽象类。

类方法：

get_strategy：

参数：无

返回：经过封装后的strategy对象

返回类型：DistributeTranspilerConfig() / DistributedStrategy()



_DistributeConfig

说明: 该类是抽象类。所有的分布式strategy配置都会继承这个抽象类。

参数：无



_ExecuteConfig

说明：该类是抽象类，gpu的执行配置都会继承这个抽象类。

参数：无



# CPU strategy:

说明：这个类是CPU分布式训练的strategy。

参数： dist_config，这个参数是一个列表，元素是关于CPU分布式配置的配置类

成员函数：

setup_optimizer

这个函数会讲CPU配置使用在optimizer中。

参数： 

fleet: 经过封装过的fleet class。

Optimizer：优化器，比如说adam

返回：经过封装后的适用于分布式策略的分布式optimizer。

 

 

SyncModeConfig：

说明：CPU分布式模式之同步模式。

参数：无

返回：无

 

HalfSyncModelConfig

说明：CPU分布式模式之半异步模式。

参数：无

返回：无

 

ASyncModelConfig：

说明：CPU分布式模式之异步模式。

参数：无

返回：无



GeoSGDModelConfig：

说明：CPU分布式模式之GEOSGD模式。

参数：need_push_nums，对应于DistributeTranspilerConfig的geo_sgd_need_push_nums参数

返回：无



# GPUStrategy：

说明：用于 GPU 的 `collective` 训练的 strategy 类。支持 paddlepaddle 的 `grad_allreduce` 与 `local_sgd` 模式。

 

成员函数

def init(self, exec_config: List[_ExecuteConfig] = [], dist_config: List[_DistributeConfig] = [])

参数:

exec_config(List[_ExecuteConfig]): 执行器配置列表

dist_config(List[_DistributeConfig]): 分布式策略配置

 

def setup_optimizer(self, optim)

参数:

optim: 用于训练的 paddlepaddle 优化器

返回:

DistributeOptimizer 对象，用于对 paddlepaddle 程序执行分布式优化

 

# ExecuteConfig

def NumThreadsConfig(num: int)

设置执行线程的数量

 

# DistributeConfig

NCCL2Mode()

使用 NCCL2 模式进行分布式训练

CollectiveMode()

使用 collective 模式训练

LocalSGD()

使用 local sgd 进行 collective 训练

GradAllreduce()

使用 grad allreduce 进行 collective 训练

HierarchicalAllreduce()

使用 hierarchical allreduce 模式