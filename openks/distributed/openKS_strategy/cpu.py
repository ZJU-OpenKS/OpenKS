# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from ..openks_strategy.base import _StrategyBase, _ExecuteConfig, _DistributeConfig


class CPUStrategy(_StrategyBase):
    def __init__(self, dist_config: list):
        print(dist_config)
        from paddle.fluid import ExecutionStrategy
        from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
        self._strategy = DistributeTranspilerConfig()
        self._strategy.wait_port = False
        for conf in dist_config:
            conf.setup(self._strategy)

    def setup_optimizer(self, fleet, optimizer):
        optimizer = fleet.distributed_optimizer(optimizer, self._strategy)
        return optimizer


class SyncModeConfig(_DistributeConfig):
    def __init__(self, ):
        super(SyncModeConfig, self).__init__()

    def setup(self, strategy):
        print(strategy)
        strategy.sync_mode = True
        strategy.runtime_split_send_recv = False


class HalfSyncModelConfig(_DistributeConfig):
    def __init__(self):
        super(HalfSyncModelConfig, self).__init__()

    def setup(self, strategy):
        strategy.sync_mode = False
        strategy.runtime_split_send_recv = False


class ASyncModelConfig(_DistributeConfig):
    def __init__(self):
        super(ASyncModelConfig, self).__init__()

    def setup(self, strategy):
        strategy.sync_mode = False
        strategy.runtime_split_send_recv = True


class GeoSGDModelConfig(_DistributeConfig):
    def __init__(self, need_push_nums):
        super(GeoSGDModelConfig, self).__init__()
        self.need_push_nums = need_push_nums

    def setup(self, strategy):
        strategy.sync_mode = False
        strategy.geo_sgd_mode = True
        strategy.geo_sgd_need_push_nums = self.need_push_nums
