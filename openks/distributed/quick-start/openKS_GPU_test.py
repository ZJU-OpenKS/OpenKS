#   Copyright (c) 2020 Room 525 Research Group, Zhejiang University.
#   All Rights Reserved.

import paddle.fluid as fluid

import os
import sys
sys.path.insert(1, os.path.dirname(__file__))
from openKS_distributed import KSDistributedFactory
from openKS_distributed.base import RoleMaker
from openKS_strategy.gpu import GPUStrategy, \
NumThreadsConfig, CollectiveMode, GradAllreduce, LocalSGD

from utils import gen_data
from nets import mlp

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)

dist_algorithm = KSDistributedFactory.instantiation(flag = 1)
role = RoleMaker.PaddleCloudRoleMaker(is_collective=True)
dist_algorithm.init(role)

# algorithm + local optimizer
optimizer = GPUStrategy(exec_config = [NumThreadsConfig(32)], dist_config = [CollectiveMode(), GradAllreduce()]).setup_optimizer(dist_algorithm, optimizer)
optimizer.minimize(cost, fluid.default_startup_program())

train_prog = dist_algorithm.main_program

gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
place = fluid.CUDAPlace(gpu_id)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

step = 1001
for i in range(step):
    cost_val = exe.run(
        program=train_prog,
        feed=gen_data(),
        fetch_list=[cost.name])
    print("worker_index: %d, step%d cost = %f" %
          (dist_algorithm.worker_index(), i, cost_val[0]))
