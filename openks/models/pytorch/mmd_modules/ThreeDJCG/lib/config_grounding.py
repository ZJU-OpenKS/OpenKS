import os
import sys
from easydict import EasyDict
from .config import CONF as CONF_BASE
import copy

CONF = copy.deepcopy(CONF_BASE)
print('Using Grounding Config')

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs/exp_grounding")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126
CONF.TRAIN.SEED = 42
