from openks.loaders import *
from openks.models import *
from openks.app.qa import *

# test loader
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENBASE
#loader_config.source_uris = ['openks/data/ent_test1.csv', 'openks/data/ent_test2.csv', 'openks/data/rel_test.csv']
#loader_config.source_uris = 'openks/data/investor-company-patent.zip'
#loader_config.source_uris = 'openks/data/wiki-covid-19-v0.3.json'
loader_config.source_uris = 'openks/data/openbase-legalkg'
loader_config.data_name = 'wiki-covid-19'
#loader_config.ent_rel_mapping = {'company_investor': {'from': {'investor': 'investor_id'}, 'to': {'company': 'company_id'}}, 'company_patent': {'from': {'company': 'company_id'}, 'to': {'patent': 'patent_id'}}}
loader = Loader(loader_config)
#print(mdd.headers)
#for body in mdd.bodies:
#	for line in body:
#		print(line)
#		break

# test graph loader
graph_loader = GraphLoader(loader_config)

print("-----------------------------------------------")
print("Dataset headers:")
#print(graph_loader.dataset.headers[0])
print("Dataset bodies:")
#print(graph_loader.dataset.bodies[0][:10])


graph = graph_loader.graph
print("-----------------------------------------------")
print("The loaded graph infomation: ")
print("Graph name: ")
print(graph.graph_name)
print("Graph schema: ")
print(graph.schema)
print("Graph entities: ")
print(graph.entities.keys())
for k in graph.entities.keys():
	print(graph.entities[k]['pointer'])
	print(graph.entities[k]['instances'][:5])

print("Graph relations: ")
print(graph.relations.keys())
for k in graph.relations.keys():
	print(graph.relations[k]['pointer'])
	print(graph.relations[k]['instances'][:5])
print("-----------------------------------------------")
print("")
# test simple model
#simple_model = SimpleModel()
print(KSPaddleModel.list_modules())

# test question parser
question = input("输入问题：")
parser = RuleParserCom(question, graph)
struc_q = parser.parse()
print("-----------------------------------------------")
print("Question: " + struc_q.text)
print("Structured question through parser: ")
print(struc_q.entities)
print(struc_q.relations)
print(struc_q.target_type)
print(struc_q.question_class)
print("")
fetcher = AnswerFetcher(struc_q, graph)
print("Answers: ")
print(fetcher.fetch_by_one_hop())
print("-----------------------------------------------")
