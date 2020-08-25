from openks.loaders import loader_config, SourceType, FileType, GraphLoader
from openks.apps.qa import RuleParserCom, AnswerFetcher
from py2neo import Graph

''' 图谱载入与图谱数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
# loader_config.source_type = SourceType.NEO4J
# graph_db = Graph(host='127.0.0.1', http_port=7474, user='neo4j', password='123456')
# loader_config.graph_db = graph_db
loader_config.file_type = FileType.OPENKS
loader_config.source_uris = 'openks/data/company-kg'
# loader_config.source_uris = 'openks/data/medical-kg'
loader_config.data_name = 'my-data-set'
# 图谱数据结构载入
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
# 图谱图数据库写入
print("将graph导入数据库：")
# 将MTG图写入图数据库，clean为False表示不进行清空和重新导入
# graph_loader.graph2neo(graph, graph_db, clean=False)

''' 知识图谱问答 '''
# 选择问题解析类并进行模型预加载
parser = RuleParserCom(graph)
# parser = RuleParserMedical(graph)
while(1):
	question = input("输入问题：")
	struc_q = parser.parse(question)
	# 进行答案获取
	fetcher = AnswerFetcher(struc_q)
	print("答案：")
	# 选择答案获取的方式
	print(fetcher.fetch_by_matching(graph))
	# print(fetcher.fetch_by_db_query(graph_db))
	print("-----------------------------------------------")