from openks.loaders import *
from openks.models import *
from openks.app.qa import *
from py2neo import Graph

''' 图谱载入与图谱数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
loader_config.source_uris = 'openks/data/medical'
loader_config.data_name = 'test-data-set'
loader = Loader(loader_config)
dataset = loader.dataset
dataset.info_display()
# 图谱数据结构载入
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
# 图谱图数据库写入
print("将graph导入数据库：")
graph_db = Graph(host='127.0.0.1', http_port=7474, user='neo4j', password='123456')
graph_loader.graph2neo(graph, graph_db, clean=True)

''' 图谱表示学习模型训练 '''
# 列出已加载模型
# OpenKSModel.list_modules()
# 算法模型选择配置
# platform = 'PyTorch'
# model_type = 'KGLearn'
# model = 'TransE'
# print("根据配置，使用 {} 框架，{} 类型的 {} 模型。".format(platform, model_type, model))
# print("-----------------------------------------------")
# 模型训练
# model_type = OpenKSModel.get_module(platform, model_type)
# kgmodel = model_type(graph=graph, model=OpenKSModel.get_module(platform, model), args=None)
# kgmodel.run()
# print()

''' 知识图谱问答 '''
while(1):
	question = input("输入问题：")
	# 选择问题解析类
	# parser = RuleParserCom(question, graph)
	parser = RuleParserMedical(question, graph)
	struc_q = parser.parse()
	# 进行答案获取
	# fetcher = AnswerFetcher(struc_q, graph)
	fetcher = AnswerFetcher(struc_q, graph)
	print("答案：")
	# print(fetcher.fetch_by_matching())
	print(fetcher.fetch_by_db_query(graph_db))
	print("-----------------------------------------------")
