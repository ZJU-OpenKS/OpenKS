# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from .gpu import algorithm as heterogeneous_algorithm
from .cpu import algorithm as general_algorithm

class KSDistributedFactory:
    @staticmethod
    def instantiation(flag):
        return heterogeneous_algorithm if flag else general_algorithm

