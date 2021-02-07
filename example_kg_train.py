# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from openks.loaders import loader_config, SourceType, FileType, GraphLoader
from openks.models import OpenKSModel
from py2neo import Graph

''' 图谱载入与图谱数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
# loader_config.source_type = SourceType.NEO4J
# graph_db = Graph(host='127.0.0.1', http_port=7474, user='neo4j', password='123456')
# loader_config.graph_db = graph_db
# loader_config.source_uris = 'openks/data/company-kg'
loader_config.source_uris = 'openks/data/FB15k-237'
# loader_config.source_uris = 'openks/data/medical-kg'
loader_config.data_name = 'my-data-set'
# 图谱数据结构载入
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
''' 图谱表示学习模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
# 算法模型选择配置
args = {
	'gpu': True,
	'learning_rate': 0.001, 
	'epoch': 500,
	'batch_size': 1024, 
	'optimizer': 'adam', 
	'hidden_size': 500, 
	'margin': 4.0, 
	'model_dir': './',
	'eval_freq': 20,
	'gamma': 12.0,
	'epsilon': 2.0
}
platform = 'PyTorch'
executor = 'KGLearn'
model = 'RotatE'
# model = 'TransE'
args['model_dir'] = model+'.pt'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
kglearn = executor(graph=graph, model=OpenKSModel.get_module(platform, model), args=args)
kglearn.run()
print("-----------------------------------------------")
