import os


class DATA_Config(object):
    train_dev_test = [8, 8, 2]
    DATA_PATH = "./data/"
    TRAIN_PATH = os.path.join(DATA_PATH, "train0.txt")
    DEV_PATH = os.path.join(DATA_PATH, "dev0.txt")
    TEST_PATH = os.path.join(DATA_PATH, "test0.txt")
    RESULT_TXT = os.path.join(DATA_PATH, "PRF10.txt")
