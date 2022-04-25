import json
import os
DIR_PATH = os.getcwd()
import sys
import xlrd
import random
import torch
import requests


import re
import urllib.parse, urllib.request
import hashlib
import urllib
import random
import json
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
FILE_PATH = os.path.join(DIR_PATH)
RESULT_PATH = os.path.join(FILE_PATH, 'entity_image.xlsx')
IMAGE_FILE_PATH = os.path.join(FILE_PATH, 'images')


class mydataloader():
    def __init__(self,path):

        x,y,o,p = self.read_datas(path)
        self.train = x
        self.test = y
        self.train_list = o
        self.test_list = p
        print("数据加载完毕")
        print("训练集train的规模为： " + str(len(self.train)))
        print("测试集test的规模为： " + str(len(self.test)))

    def read_datas(self,path):
        train = {}
        test = {}
        train_list = []
        test_list = []

        """
        f1 = open("processed_train2.json", 'r', encoding="utf-8")
        data_train = json.load(f1)
        f2 = open("processed_test2.json", 'r', encoding="utf-8")
        data_test = json.load(f2)
        for shiti_train in data_train:
            train[shiti_train] = data_train[shiti_train]
            train_list.append(shiti_train)
        for shiti_test in data_test:
            test[shiti_test] = data_test[shiti_test]
            test_list.append(shiti_test)
        """

        f = open(path, 'r', encoding="utf-8")
        datas = json.load(f)
        x_len = len(datas)
        i = 0
        for shiti in datas:
            if i < int(x_len * 0.9):
            #if i < 400:
                train[shiti] = datas[shiti]
                train_list.append(shiti)
            else:
                test[shiti] = datas[shiti]
                test_list.append(shiti)
            i = i + 1

        return train,test,train_list,test_list

