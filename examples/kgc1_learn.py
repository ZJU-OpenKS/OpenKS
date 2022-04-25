# -*- coding: utf-8 -*-
# @Time    : 2021/7/05 10:39
# @Author  : Benjamin
# @FileName: kgc1_learn.py

from openks.loaders import loader_config, SourceType, FileType, GraphLoader
from openks.models import OpenKSModel
from py2neo import Graph
import os

''' 图谱载入与图谱数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
# loader_config.file_type = FileType.OPENKS
# # loader_config.source_type = SourceType.NEO4J
# # graph_db = Graph(host='127.0.0.1', http_port=7474, user='neo4j', password='123456')
# # loader_config.graph_db = graph_db
# loader_config.source_uris = 'openks/data/company-kg'
# # loader_config.source_uris = 'openks/data/medical-kg'
# loader_config.data_name = 'my-data-set'
# # 图谱数据结构载入
# graph_loader = GraphLoader(loader_config)
# graph = graph_loader.graph
# graph.info_display()
graph = None
''' 图谱表示学习模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
# 算法模型选择配置
args = {
	'gpu': False,
	'learning_rate': 0.003,
	# 'epoch': 10,
	'batch_size': 32,
	# 'optimizer': 'adam',
	# 'hidden_size': 50,
	# 'margin': 4.0,
	'data_dir': './openks/data/FB15k',
	'model_dir': './',
	# 'eval_freq': 1,

    "num_iterations":500,
    "dr":0.99,
    "edim":200,
    "rdim":200,
    "input_dropout":0.2,
    "hidden_dropout1":0.2,
    "hidden_dropout2":0.3,
    "label_smoothing":0.,
    # 训练+评估
    "test_mode":False,
    "cuda":True,
}
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
platform = 'PyTorch'
executor = 'KGC1Learn'
model = 'KGC1'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
kgc1learn = executor(graph=graph, model=OpenKSModel.get_module(platform, model), args=args)
kgc1learn.run()
print("-----------------------------------------------")
