# -*- coding: utf-8 -*-
#
# Copyright 2022 HangZhou Hikvision Digital Technology Co., Ltd. All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import importlib
import sys


class ModelApiError(Exception):
    """Model api error"""


def get_logger(verbosity_level, name, use_error_log=False):
    """Set logging format to something
    """
    logger = logging.getLogger(name)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)
METHOD_LIST = ['train_predict']


def _check_umodel_methed(umodel):
    """Check if the model has methods in METHOD_LIST"""
    for attr in ['train_predict']:
        if not hasattr(umodel, attr):
            raise ModelApiError(
                f"Your model object doesn't have the method attr")


def import_umodel():
    """import user model"""
    model_cls = importlib.import_module('model').Model
    _check_umodel_methed(model_cls)

    return model_cls


def init_usermodel():
    """initialize user model"""
    return import_umodel()()
