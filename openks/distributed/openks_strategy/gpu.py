# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from ..openks_strategy.base import _StrategyBase, _ExecuteConfig, _DistributeConfig
from typing import List


class GPUStrategy(_StrategyBase):
    def __init__(self, exec_config: List[_ExecuteConfig] = [], dist_config: List[_DistributeConfig] = []):
        from paddle.fluid import ExecutionStrategy
        from paddle.fluid.incubate.fleet.collective import DistributedStrategy
        self._strategy = DistributedStrategy()
        for conf in dist_config:
            conf.setup(self._strategy)
        for conf in exec_config:
            conf.setup(self._strategy.exec_strategy)

    def setup_optimizer(self, fleet, optim):
        return fleet.distributed_optimizer(optim, strategy=self._strategy)


class NumThreadsConfig(_ExecuteConfig):
    def __init__(self, num: int):
        self.num = num

    def setup(self, strategy):
        strategy.num_threads = self.num


class NCCL2Mode(_DistributeConfig):
    def setup(self, s):
        s.mode = 'nccl2'


class LocalSGD(_DistributeConfig):
    def setup(self, s):
        s.collective_mode = 'local_sgd'


class GradAllreduce(_DistributeConfig):
    def setup(self, s):
        s.collective_mode = 'grad_allreduce'


class CollectiveMode(_DistributeConfig):
    def setup(self, s):
        s.mode = 'collective'


class HierarchicalAllreduce(_DistributeConfig):
    def setup(self, strategy):
        strategy.use_hierarchical_allreduce = True

