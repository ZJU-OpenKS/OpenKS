import logging
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from ..model import VisualConstructionModel
from .visual_entity_modules import clip
from .visual_entity_modules.datasets import loaddata
from .visual_entity_modules.newbert_model import TransformerBiaffine as Model
import argparse


@VisualConstructionModel.register("VisualEntityExtract", "PyTorch")
class VisualEntityExtractTorch(VisualConstructionModel):
    def __init__(self, name: str, dataset=None, args=None):
        # super().__init__(name=name, dataset=dataset, args=args)
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--seed', type=int, default=19980524)
        parser.add_argument('--dropout', type=float, default=0.1)
        args1 = parser.parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = loaddata.mydataloader("processed_train3.json")
        self.traindata = self.env.train
        self.testdata = self.env.test
        self.train_list = self.env.train_list
        self.test_list = self.env.test_list
        self.model = Model(args1)

    def parse_args(self,args):
        return args

    # def data_reader(self, *args):

    #    return super().data_reader(*args)

    def evaluate(self, *args):
        print("开始测试啦！")
        i = 0
        self.model.eval()
        batch_size = 1
        total = 0
        m_right = 0
        with torch.no_grad():
            while True:
                # data_shiti = torch.zeros((batch_size, 1, 77)).to(self.device)
                data_tupian = torch.zeros((batch_size, 20, 512)).to(self.device)  # 正例和负例最多是20个
                shiti = []
                # label = torch.zeros((batch_size,20)).to(device)
                # mask = torch.zeros((batch_size,20)).to(device)
                label2 = []
                for o in range(0, 1):
                    label2.append([])
                    for p in range(0, 20):
                        label2[o].append(0)
                # mask2 = [[False] * 20] * batch_size
                mask2 = []
                for o in range(0, 1):
                    mask2.append([])
                    for p in range(0, 20):
                        mask2[o].append(False)
                m = self.testdata[self.test_list[i]]["正例个数"]
                n = self.testdata[self.test_list[i]]["负例个数"]
                # data_shiti[0] = torch.tensor(self.testdata[test_list[i]]["实体向量"]).to(self.device)
                shiti.append(self.testdata[self.test_list[i]]["英文名"])
                for k in range(0, (m + n)):
                    if k < m:
                        label2[0][k] = 1
                        mask2[0][k] = True
                        data_tupian[0][k] = torch.tensor(self.testdata[self.test_list[i]]["图片向量"][k]).to(self.device)
                    else:
                        label2[0][k] = 0
                        mask2[0][k] = True
                        data_tupian[0][k] = torch.tensor(self.testdata[self.test_list[i]]["图片向量"][k]).to(self.device)
                label = torch.tensor(label2).to(self.device)
                mask = torch.tensor(mask2).to(self.device)
                # data_shiti = data_shiti.type(torch.long)
                _, muls = self.model(shiti, None, data_tupian, None, None)

                # muls = torch.sigmoid(muls)  ###########
                total = total + m
                j = 0
                for gailv in muls[0]:
                    xxxx = torch.argmax(gailv).item()
                    if xxxx == 1:
                        m_right = m_right + 1
                    j = j + 1
                    if j >= m:
                        break
                i = i + 1
                if i == len(self.test_list):
                    break

        label1 = m_right / total
        print("测试的结果是" + str(label1))
        return label1

    def train(self, *args):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        maxlabel1 = 0
        batch_size = 4
        for epoch in range(0, 50):
            self.model.train()
            print("训练到第 " + str(epoch) + "轮了")
            counter = 0
            stop = False
            while True:

                if counter % 200 == 0:
                    print("epoch :  " + str(epoch) + " . 正在训练第 " + str(counter) + " 条数据")
                shiti_num = counter * batch_size  # 轮到第几个实体来输入模型了
                shiti = []
                if shiti_num + batch_size < len(self.train_list):
                    # 构造batch_size大小个数据
                    # data_shiti = torch.zeros((batch_size, 1, 77)).to(self.device)
                    data_tupian = torch.zeros((batch_size, 20, 512)).to(self.device)  # 正例和负例最多是20个
                    # label = torch.zeros((batch_size,20)).to(device)
                    # mask = torch.zeros((batch_size,20)).to(device)
                    # label2 = [[0] * 20 ]* batch_size
                    label2 = []
                    for o in range(0, batch_size):
                        label2.append([])
                        for p in range(0, 20):
                            label2[o].append(0)
                    # mask2 = [[False] * 20] * batch_size
                    mask2 = []
                    for o in range(0, batch_size):
                        mask2.append([])
                        for p in range(0, 20):
                            mask2[o].append(False)
                    for j in range(shiti_num, shiti_num + batch_size):
                        m = self.traindata[self.train_list[j]]["正例个数"]
                        n = self.traindata[self.train_list[j]]["负例个数"]
                        # data_shiti[j - shiti_num] = torch.tensor(self.traindata[self.train_list[j]]["实体向量"]).to(self.device)
                        shiti.append(self.traindata[self.train_list[j]]["英文名"])
                        for k in range(0, (m + n)):
                            if k < m:
                                label2[j - shiti_num][k] = 1
                                mask2[j - shiti_num][k] = True
                                data_tupian[j - shiti_num][k] = torch.tensor(
                                    self.traindata[self.train_list[j]]["图片向量"][k]).to(
                                    self.device)
                            else:
                                label2[j - shiti_num][k] = 0
                                mask2[j - shiti_num][k] = True
                                data_tupian[j - shiti_num][k] = torch.tensor(
                                    self.traindata[self.train_list[j]]["图片向量"][k]).to(
                                    self.device)
                else:
                    stop = True
                    batch_size1 = len(self.train_list) - shiti_num
                    # data_shiti = torch.zeros((batch_size1, 1, 77)).to(self.device)
                    data_tupian = torch.zeros((batch_size1, 20, 512)).to(self.device)  # 正例和负例最多是20个
                    # label = torch.zeros((batch_size1, 20)).to(device)
                    # mask = torch.zeros((batch_size1, 20)).to(device)
                    # label2 = [[0] * 20] * batch_size1
                    # mask2 = [[False] * 20] * batch_size1
                    # label2 = [[0] * 20 ]* batch_size
                    label2 = []
                    for o in range(0, batch_size1):
                        label2.append([])
                        for p in range(0, 20):
                            label2[o].append(0)
                    # mask2 = [[False] * 20] * batch_size
                    mask2 = []
                    for o in range(0, batch_size1):
                        mask2.append([])
                        for p in range(0, 20):
                            mask2[o].append(False)
                    for j in range(shiti_num, shiti_num + batch_size1):
                        m = self.traindata[self.train_list[j]]["正例个数"]
                        n = self.traindata[self.train_list[j]]["负例个数"]
                        # data_shiti[j - shiti_num] = torch.tensor(self.traindata[self.train_list[j]]["实体向量"]).to(self.device)
                        shiti.append(self.traindata[self.train_list[j]]["英文名"])
                        for k in range(0, (m + n)):
                            if k < m:
                                label2[j - shiti_num][k] = 1
                                mask2[j - shiti_num][k] = True
                                data_tupian[j - shiti_num][k] = torch.tensor(
                                    self.traindata[self.train_list[j]]["图片向量"][k]).to(
                                    self.device)
                            else:
                                label2[j - shiti_num][k] = 0
                                mask2[j - shiti_num][k] = True
                                data_tupian[j - shiti_num][k] = torch.tensor(
                                    self.traindata[self.train_list[j]]["图片向量"][k]).to(
                                    self.device)
                label = torch.tensor(label2).to(self.device)
                mask = torch.tensor(mask2).to(self.device)
                # data_shiti = data_shiti.type(torch.long)
                loss, _ = self.model(shiti, None, data_tupian, label, mask)
                # print(loss)

                if counter % 200 == 0:
                    print("print loss")
                    print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter = counter + 1
                if stop:
                    optimizer.step()  # 更新参数
                    optimizer.zero_grad()  # 梯度清零
                    print("shiti_num")
                    print(shiti_num)
                    print("loss  :  ")
                    print(loss)
                    break
            try:
                label1 = self.evaluate()
            except:
                print("测试函数出错")

            # pposgd_simple_gcn.the_test(args, env, policy_fn)  ##########测试程序
            if label1 > maxlabel1:
                self.evaluate()  #################再测一遍，主要目的是存模型
                print("存储模型##################################")
                modelpath = "./visual_entity_model.pth"
                torch.save(self.model.state_dict(), modelpath)
                print("模型存储完毕!")
                maxlabel1 = label1

    def run(self, mode="train"):
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.evaluate()
        elif mode == "single":
            raise ValueError("UnImplemented mode!")