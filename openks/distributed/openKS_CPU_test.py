#   Copyright (c) 2020 Room 525 Research Group, Zhejiang University.
#   All Rights Reserved.

import paddle.fluid as fluid

import os
import sys
sys.path.insert(1, os.path.dirname(__file__))
import numpy as np
from openKS_distributed import KSDistributedFactory
from openKS_distributed.base import RoleMaker
from openKS_strategy.cpu import CPUStrategy, SyncModeConfig

def mlp(input_x, input_y, hid_dim=128, label_dim=2):
    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
    prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost

def gen_data():
    return {"x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64')}

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

