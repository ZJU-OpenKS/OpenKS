#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from functools import reduce
import networkx as nx

#类说明
# 整体方法有一个共同的输入数据，即投资高管关系数据，具体可见数据文件
# 实现了三个方法：
# 1 控股关系识别holding_invest(person_or_enterprise,deep_num,ratio_p)
#     参数说明：person_or_enterprise：string类型，输入枚举（’person‘）个人或（’enterprise‘）企业，返回个人控股关系或企业控股关系；
#             deep_num：int类型，用于控制控股关系的最大穿透层数；
#             ratio_p：double类型，用于筛选控股关系的控股比例阈值
#     返回值：dict类型，key为主控人（法人），value为dict类型，包含被控企业以及控股比例
# 2 一致行动人识别concerted_action_people()
#     返回值：tuple类型，列出了三类（总监高，个人直接持股30%以上，公用董监高三种一致行动人结果）
# 3 实控人识别actual_controller()
#     无需输入参数
#     返回所有实控人dict，key为公司，value为实控人


class financial_risk_control_model:
    #数据初始化
    def __init__(self,filename):
        self.data = pd.read_csv(filename)

    #获取投资对象数据，即在有向图中为邻居
    def get_all_neighbors(self):
        data = self.data
        result = data.groupby('src_uid')["dst_uid"].apply(list).to_dict()
        return result
    
    #获取所有投资路径
    def findAllPath(self,nei_graph,start,end,deep_num,path=[]):
        path = path +[start]
        if start == end:
            return [path]
        paths = [] #存储全部路径    
        for node in nei_graph[start]:
            if node not in path:
                newpaths = self.findAllPath(nei_graph,node,end,deep_num,path) 
                for newpath in newpaths:
                    if len(newpath)<=deep_num:
                        paths.append(newpath)
        return paths
    
    #计算投资比例
    def comp_invest_ratio(self,all_path):
        if len(all_path)==0:
            return 0
        inv = []
        for i in all_path:        
            if len(i)>2:
                invest_list = []
                invest_list_ratio = 0
                for j in [(x,y) for x, y in zip(i, i[1:])]:                   
                    invest_r = self.data[(self.data['src_uid']==j[0])&(self.data['dst_uid']==j[1])]
                    if len(invest_r)>0:
                        invest_list.append(invest_r.invest_ratio.values[0])                       
                if len(invest_list)>1:
                    invest_list_ratio = reduce(lambda x, y: x*y, invest_list)
                if len(invest_list)==1:
                    invest_list_ratio = invest_list[0]
                inv.append(invest_list_ratio)
            if len(i)==2:
                invest_r = self.data[(self.data['src_uid']==i[0])&(self.data['dst_uid']==i[1])].invest_ratio.values
                if len(invest_r)>0:
                    inv.append(invest_r[0])
        return sum(inv)
    
    #计算所有投资比例之和    
    def find_all_invest(self,start_company,deep_num,ratio_p):
        if start_company not in self.data.src_uid.values:
            return None
        data = self.data
        result = {}
        graph = self.get_all_neighbors()
        for i in data.src_uid.values:
            if i!=start_company:
                all_path = self.findAllPath(graph,start_company,i,deep_num,path=[])
                start_company_inv_ratio = self.comp_invest_ratio(all_path)
                if start_company_inv_ratio>ratio_p:
                    result[i] = start_company_inv_ratio
        return result
    
    #计算数据中的所有控股信息
    def holding_invest(self,person_or_enterprise,deep_num,ratio_p):
        data = self.data
        data = data[data.src_label==person_or_enterprise]
        result = {}
        for i in set(data.src_uid.values):
            k = self.find_all_invest(i,deep_num,ratio_p)
            if len(k)>0:
                result[i] = k
        return result
    
    #一致行动人识别,返回##总监高，##个人直接持股30%以上，##公用董监高三种一致行动人结果
    def concerted_action_people(self):
        data = self.data
        data1 = data[(data.edge_label=='officer')&(data.position.isnull()==False)][['src_uid','dst_uid']]
        data2 = data[(data.invest_ratio>0.3)&(data.src_label=='person')&(data.edge_label=='invest')][['src_uid','dst_uid']]   
        data3 = data1.groupby("src_uid")["dst_uid"].apply(list).values                                  
        return (data1.values,data2.values,data3)
    
    #实际控制人识别
    def actual_controller(self):
        edges = []
        h = self.holding_invest('person',3,0.5)
        for i in h.keys():
            for j in h[i]:
                edges.append((i,j))
        G = nx.DiGraph()
        G.add_edges_from(edges)
        result = {}
        for i in G.nodes():
            con_i = []
            n = [i]
            while True:
                n = list(G.predecessors(n[0]))
                if len(n) != 0:
                    con_i.append(n[0])
                else:
                    break
            if len(con_i)>0:
                result[i] = con_i[-1]
        return result

if __name__ == "__main__":
    t = financial_risk_control_model('openks_finance_data.csv')
    print(t.concerted_action_people())
    print(t.holding_invest('person',3,0.4))
    print(t.holding_invest('enterprise',3,0.4))
    print(t.actual_controller())

