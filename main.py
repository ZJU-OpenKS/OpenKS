from openks.loaders import *
from openks.models import *
from openks.app.qa import *

# test loader
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.CSV
#loader_config.source_uris = ['openks/data/ent_test1.csv', 'openks/data/ent_test2.csv', 'openks/data/rel_test.csv']
loader_config.source_uris = 'openks/data/investor-company-patent.zip'
#loader_config.source_uris = 'openks/data/wiki-covid-19-v0.3.json'
#loader_config.source_uris = 'openks/data/openbase-wiki'
loader_config.data_name = 'wiki-covid-19'
loader = Loader(loader_config)
dataset = loader.dataset
dataset.info_display()

graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
# test simple model
#simple_model = SimpleModel()
print(KSPaddleModel.list_modules())

# test question parser
question = input("输入问题：")
parser = RuleParserCom(question, graph)
struc_q = parser.parse()
fetcher = AnswerFetcher(struc_q, graph)
print("答案：")
print(fetcher.fetch_by_matching())
print("-----------------------------------------------")
