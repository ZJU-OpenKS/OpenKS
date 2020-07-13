from openks.loaders import *
from openks.models import *
from openks.models.paddle import *
from openks.app.qa import *

# test loader
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
loader_config.source_uris = 'openks/data/company-dev'
loader_config.data_name = 'test-data-set'
loader = Loader(loader_config)
dataset = loader.dataset
dataset.info_display()

graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
# test model
#simple_model = SimpleModel()
print(KGModel.list_modules())
kgmodel = KGModel(graph=graph, model='TransE', args=None)
kgmodel.run()
print()
# test question parser
question = input("输入问题：")
parser = RuleParserCom(question, graph)
struc_q = parser.parse()
fetcher = AnswerFetcher(struc_q, graph)
print("答案：")
print(fetcher.fetch_by_matching())
print("-----------------------------------------------")
