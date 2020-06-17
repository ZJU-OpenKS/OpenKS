from openks.loaders import *
from openks.models import *
from openks.app.qa import *

# test loader
loader_config.source_type = SourceType.LOCAL_FILE
#loader_config.source_uris = ['openks/data/ent_test1.csv', 'openks/data/ent_test2.csv', 'openks/data/rel_test.csv']
loader_config.source_uris = 'openks/data/investor-company-patent.zip'
loader_config.data_name = 'test'
loader_config.ent_rel_mapping = {'company_investor': {'from': 'investor', 'to': 'company', 'from_attr': 'investor_id', 'to_attr':'company_id'}, 'company_patent': {'from': 'company', 'to': 'patent', 'from_attr': 'company_id', 'to_attr':'patent_id'}}
loader = Loader(loader_config)
#print(mdd.headers)
#for body in mdd.bodies:
#	for line in body:
#		print(line)
#		break

# test graph loader
loader_config.data_name = 'default-graph'
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
print("-----------------------------------------------")
print("The loaded graph infomation: ")
print("Graph name: ")
print(graph.graph_name)
print("Graph entity types: ")
print(graph.entity_types)
print("Graph relation types: ")
print(graph.relation_types)
print("Graph entity attributes: ")
print(graph.entity_attrs)
print("Graph relation attributes: ")
print(graph.relation_attrs)
print("-----------------------------------------------")
print("")
# test simple model
#simple_model = SimpleModel()
print(KSModel.list_modules())

# test question parser
parser = RuleParserCom("奇安信申请了几个专利？", graph)
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
print(fetcher.fetch_by_one())
print("-----------------------------------------------")
