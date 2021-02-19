# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import os, argparse
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
dataset_name = 'FB15k-237'
loader_config.source_uris = 'openks/data/'+dataset_name
# loader_config.source_uris = 'openks/data/medical-kg'
loader_config.data_name = 'my-data-set'
# 图谱数据结构载入
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
''' 图谱表示学习模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()


def parse_args(args=None):
	parser = argparse.ArgumentParser(
		description='Training and Testing Knowledge Graph Embedding Models',
		usage='train.py [<args>] [-h | --help]'
	)
	parser.add_argument('--model', default='TransE', type=str)
	parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
	parser.add_argument('--max_steps', default=100000, type=int)
	parser.add_argument('-ef', '--eval_freq', default=10000, type=int)
	parser.add_argument('-de', '--double_entity_embedding', action='store_true')
	parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

	return parser.parse_args(args)


args_from_parse = parse_args()
# 算法模型选择配置
args = {
	'gpu': True,
	'learning_rate': 0.00005,
	'epoch': 1024,
	'batch_size': 1024, 
	'optimizer': 'adam',
	'margin': 4.0,
	'data_dir': loader_config.source_uris,
	'log_steps': 100,
	'test_log_steps': 1000,
	'gamma': 9.0,
	'epsilon': 2.0,
	'negative_sample_size': 256,
	'test_batch_size': 16,
	'negative_adversarial_sampling': True,
	'adversarial_temperature': 1.0,
	'cpu_num': 10,
	'warm_up_steps': None,
	'init_checkpoint': None,
	'uni_weight': False,
	'regularization': 0.0,
	'do_valid': True,
	'do_test': True,
	'evaluate_train': True,
	'random_split': False,
	'random_seed': 1
}
platform = 'PyTorch'
executor = 'KGLearn'
# model = 'TransE'
model = args_from_parse.model
args['model_name'] = model
args['hidden_size'] = args_from_parse.hidden_dim
args['max_steps'] = args_from_parse.max_steps
args['eval_freq'] = args_from_parse.eval_freq
args['model_dir'] = 'models/'+model+'_'+dataset_name+'_'+str(args['random_seed'])
args['save_path'] = args['model_dir']
args['double_entity_embedding'] = args_from_parse.double_entity_embedding
args['double_relation_embedding'] = args_from_parse.double_relation_embedding
if not os.path.exists(args['save_path']):
	os.makedirs(args['save_path'])
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
kglearn = executor(graph=graph, model=OpenKSModel.get_module(platform, model), args=args)
kglearn.run()
print("-----------------------------------------------")
