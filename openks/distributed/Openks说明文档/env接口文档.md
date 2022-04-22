### Class ps_launcher(object):

说明：ps_launcher类用于使用cpu进行Parameter Server模式进行训练时，对环境进行配置

**函数：**

def  init(self)

创建ps_launcher对象，调用该类的静态方法parse_args对节点ip地址，sever节点数量，worker节点数量进行 等参数进行设置。

参数：无

返回：无 

返回类型：无



def parse_args()

对节点ip地址，开始端口号，sever节点数量，worker节点数量，输出日志路径，训练脚本路径等参数进行设置。参数由用户在终端运行训练脚本时，使用环境变量指定。部分参数带有默认值，sever节点数量和worker节点数量必须由用户指定。

参数：无

返回：argparse.ArgumentParser实例

返回类型：argparse.ArgumentParser



def start_procs(self)

根据环境配置，启动Parameter Server模式进行CPU下分布式训练

参数：无

返回：无

返回类型：无



def launch(self)

调用该类的实例函数start_procs，开始分布式训练

参数：无

返回：无

返回类型：无



### Class launcher(object):

说明：launcher类用于使用gpu进行Parameter Server模式进行训练时，对环境进行配置。使用该训练器时，训练program必须在nccl2模式下运行，并且在不同进程初始化中需要正确的按顺序读取以下环境变量：

FLAGS_selected_gpus

​	PADDLE_TRAINER_ID

​    PADDLE_CURRENT_ENDPOINT

​    PADDLE_TRAINERS_NUM

​    PADDLE_TRAINER_ENDPOINTS

​    POD_IP (current node ip address, not needed for local training



**函数：**

def  init(self)

创建launcher对象，调用该类的静态方法parse_args对节点ip地址，sever节点数量，worker节点数量进行 等参数进行设置。

参数：无

返回：无

返回类型：无



def parse_args()

对节点ip地址，开始端口号，sever节点数量，worker节点数量，输出日志路径，训练脚本路径等参数进行设置。参数由用户在终端运行训练脚本时，使用环境变量指定。部分参数带有默认值，sever节点数量和worker节点数量必须由用户指定。

参数：无

返回：argparse.ArgumentParser实例

返回类型：argparse.ArgumentParser

 

def print_argument()

打印命令行参数变量



def get_cluster_from_args(selected_gpus)

将节点ip与端口组合成套接字，利用各节点地址创界集群Cluster类对象。

参数：

selected_gpus 选中的gpu编号列表

返回：

集群Cluster类对象

 

def get_gpus(selected_gpus)

根据selected_gpu返回选中的gpu的编号列表。

例如环境变量为CUDA_VISIBLE_DEVICES=4,5,6,7时，若selected_gpus=4,5,6,7，

则返回的selected_gpus=0,1,2,3

参数

selected_gpus 需要选中的gpu在环境变量里的序号的列表

返回：

选中的gpu编号列表

 

def launch(self)

创建进程，将trainging_script程序部署到集群的各个设备上。

参数：无

返回：无

 

### Class openKS_launcher(object):

**类说明：**

openKS框架下用于进行分布式环境配置的类


**函数：**

def  init(self, mode)

根据mode参数，实例化cpu下Parameter Server模式的ps_launcher类或gpu下Collective模式的launcher类，作为openKS_launcher类的launcher。

参数：mode，字符串参数，用于判断使用cpu进行分布式训练，还是使用gpu进行分布式训练

返回：无 

返回类型：无

 

def launch(self):

调用self.launcher的launch方法，开始对应模式的分布式训练。

参数：无 

返回：无

返回类型：无

 

