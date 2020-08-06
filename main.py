from openks.loaders import *
from openks.models import *
from openks.apps.qa import *
from py2neo import Graph

''' 图谱载入与图谱数据结构生成 '''
# 载入参数配置与数据集载入
# loader_config.source_type = SourceType.LOCAL_FILE
loader_config.source_type = SourceType.NEO4J
graph_db = Graph(host='127.0.0.1', http_port=7474, user='neo4j', password='123456')
loader_config.graph_db = graph_db
loader_config.file_type = FileType.OPENKS
# loader_config.source_uris = 'openks/data/company-dev'
loader_config.source_uris = 'openks/data/medical'
loader_config.data_name = 'test-data-set'
#loader = Loader(loader_config)
#dataset = loader.dataset
#dataset.info_display()
# 图谱数据结构载入
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
# 图谱图数据库写入
print("将graph导入数据库：")
# 将MTG图写入图数据库，clean为False表示不进行清空和重新导入
graph_loader.graph2neo(graph, graph_db, clean=False)

''' 图谱表示学习模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
# 算法模型选择配置
platform = 'Paddle'
model_type = 'KGLearn'
model = 'TransE'
print("根据配置，使用 {} 框架，{} 类型的 {} 模型。".format(platform, model_type, model))
print("-----------------------------------------------")
# 模型训练
model_type = OpenKSModel.get_module(platform, model_type)
kglearn = model_type(graph=graph, model=OpenKSModel.get_module(platform, model), args=None)
# kglearn.run()
print("-----------------------------------------------")
''' 知识图谱问答 '''
# 选择问题解析类并进行模型预加载
# parser = RuleParserCom(graph)
parser = RuleParserMedical(graph)
while(1):
	question = input("输入问题：")
	struc_q = parser.parse(question)
	# 进行答案获取
	fetcher = AnswerFetcher(struc_q, graph)
	print("答案：")
	# 选择答案获取的方式
	# print(fetcher.fetch_by_matching())
	print(fetcher.fetch_by_db_query(graph_db))
	print("-----------------------------------------------")
