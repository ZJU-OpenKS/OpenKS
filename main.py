from openks.loaders import *
from openks.models import *
from openks.app.qa import *

# test loader
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.source_uris = ['openks/data/ent_test1.csv', 'openks/data/ent_test2.csv', 'openks/data/rel_test.csv']
loader_config.data_name = 'test'
loader = Loader(loader_config)
print(mdd.headers)
for body in mdd.bodies:
    for line in body:
        print(line)

# test graph loader
loader_config.data_name = 'default-graph'
graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
print(graph)
print(graph.graph_name)
print(graph.entity_types)
print(graph.relation_types)
print(graph.entity_attrs)
print(graph.relation_attrs)
for line in graph.entities:
    print(line)
for line in graph.relations:
    print(line)

# test simple model
#simple_model = SimpleModel()
print(KSModel.list_modules())

# test question parser
parser = QuestionManager("Where are you", kg_name="kg1")
res = parser.simple_parser()
print(res.text)
print(res.entities)
print(res.relations)
