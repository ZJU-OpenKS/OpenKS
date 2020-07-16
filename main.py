from openks.loaders import *
from openks.models import *
from openks.app.qa import *

# test loader
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
loader_config.source_uris = 'openks/data/FB15k-237'
loader_config.data_name = 'test-data-set'
loader = Loader(loader_config)
dataset = loader.dataset
dataset.info_display()

graph_loader = GraphLoader(loader_config)
graph = graph_loader.graph
graph.info_display()
# test model
#simple_model = SimpleModel()
OpenKSModel.list_modules()
platform = 'Paddle'
model_type = 'KGLearn'
model = 'TransE'
print("根据配置，使用 {} 框架，{} 类型的 {} 模型。".format(platform, model_type, model))
print("-----------------------------------------------")
model_type = OpenKSModel.get_module(platform, model_type)
kgmodel = model_type(graph=graph, model=OpenKSModel.get_module(platform, model), args=None)
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
