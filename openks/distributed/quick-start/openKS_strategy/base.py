# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from abc import ABC, abstractmethod
from enum import Enum
from typing import List


"""
openKS_strategy = DistributeTranspilerConfig()
openKS_strategy.sync_mode = False
openKS_strategy.runtime_split_send_recv = True
"""


class _StrategyBase(ABC):
    _strategy = None
    def get_strategy(self):
        return self._strategy


class _DistributeConfig(ABC):
    @abstractmethod
    def setup(self, strategy):
        raise NotImplementedError()


class _ExecuteConfig(ABC):
    @abstractmethod
    def setup(self, strategy):
        raise NotImplementedError()
