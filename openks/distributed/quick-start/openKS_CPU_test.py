#   Copyright (c) 2020 Room 525 Research Group, Zhejiang University.
#   All Rights Reserved.

import paddle.fluid as fluid

import os
import sys
sys.path.insert(1, os.path.dirname(__file__))
from openKS_distributed import KSDistributedFactory
from openKS_distributed.base import RoleMaker
from openKS_strategy.cpu import CPUStrategy, SyncModeConfig

from utils import gen_data
from nets import mlp

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.Adagrad(learning_rate=0.01)

dist_algorithm = KSDistributedFactory.instantiation(flag = 0)
role = RoleMaker.PaddleCloudRoleMaker()
dist_algorithm.init(role)

# algorithm + local optimizer
optimizer = CPUStrategy([SyncModeConfig()]).setup_optimizer(dist_algorithm, optimizer)
optimizer.minimize(cost)

if dist_algorithm.is_server():
    dist_algorithm.init_server()
    dist_algorithm.run_server()
elif dist_algorithm.is_worker():
    dist_algorithm.init_worker()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    step = 1001
    for i in range(step):
        cost_val = exe.run(
            program=dist_algorithm.main_program,
            feed=gen_data(),
            fetch_list=[cost.name])
        print("worker_index: %d, step%d cost = %f" %
              (dist_algorithm.worker_index(), i, cost_val[0]))
    dist_algorithm.stop_worker()


