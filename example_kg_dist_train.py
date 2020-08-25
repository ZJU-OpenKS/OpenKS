from openks.loaders import loader_config, SourceType, FileType, GraphLoader
from openks.models import OpenKSModel

''' 图谱载入与图谱数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
# loader_config.source_uris = 'openks/data/company-kg'
loader_config.source_uris = 'openks/data/medical-kg'
loader_config.data_name = 'test-data-set'
# 图谱数据结构载入
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()

''' 图谱表示学习模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
args = {
	'gpu': False, 
	'learning_rate': 0.001, 
	'epoch': 10, 
	'batch_size': 1000, 
	'optimizer': 'adam', 
	'hidden_size': 50, 
	'margin': 4.0, 
	'model_dir': './', 
	'eval_freq': 10
}
# 算法模型选择配置
platform = 'Paddle'
model_type = 'KGLearn'
model = 'TransE'
print("根据配置，使用 {} 框架，{} 类型的 {} 模型。".format(platform, model_type, model))
print("-----------------------------------------------")
# 模型训练
model_type = OpenKSModel.get_module(platform, model_type)
kglearn = model_type(graph=graph, model=OpenKSModel.get_module(platform, model), args=args)
# 设置dist参数为True用于进行分布式训练
kglearn.run(dist=True)