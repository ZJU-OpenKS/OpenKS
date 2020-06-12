"""
Loader for generating knowledge graph data format HDG
"""
import os
import logging
from loader import Loader, LoaderConfig, SourceType
import sys
sys.path.append('..')
from abstract.hdg import HDG

logger = logging.getLogger(__name__)

loader_config = LoaderConfig()
hdg = HDG()

class GraphLoader(Loader):
	"""
	Specific loader for generating HDG format from MDD
	"""
	def __init__(
		self, 
		config: loader_config, 
		graph_name: str = ''
		) -> None:
		super(GraphLoader, self).__init__(config)
		self.graph_name = loader_config.data_name
		self.graph = self._load_data()

	def _load_data(self) -> HDG:
		""" 
		transform MDD from _read_data method to HDG. 
		We require an entity dataset has the file name likes 'ent_<entity_type>', each line in the file has at least an 'id' column. 
		We require a relation dataset has the file name likes 'rel_<relation_type>', each line in the file has at least two id columns for head entities and tail entities """
		entity_types = []
		relation_types = []
		ent_index = []
		rel_index = []
		count = 0
		file_names = [os.path.splitext(os.path.basename(path))[0] for path in self.config.source_uris]
		for name in file_names:
			if name.startswith('ent_'):
				entity_types.append(''.join(name.split('_')[1:]))
				ent_index.append(count)
			elif name.startswith('rel_'):
				relation_types.append(''.join(name.split('_')[1:]))
				rel_index.append(count)
			else:
				logging.warn("File name {} is not allowed for graph loader, should start with 'ent_' or 'rel_'".format(name))
				return None
			count += 1
		hdg.entity_types = entity_types
		hdg.relation_types = relation_types
		entity_attrs = []
		relation_attrs = []
		entities = []
		relations = []
		for ent_type, index in zip(entity_types, ent_index):
			entity_attrs.append({'type': ent_type, 'attrs': self.dataset.headers[index]})
			entities.append({'type': ent_type, 'instances': self.dataset.bodies[index]})
		for rel_type, index in zip(relation_types, rel_index):
			relation_attrs.append({'type': rel_type, 'attrs': self.dataset.headers[index]})
			relations.append({'type': rel_type, 'instances': self.dataset.bodies[index]})
		hdg.entity_attrs = entity_attrs
		hdg.relation_attrs = relation_attrs
		hdg.entities = entities
		hdg.relations = relations
		hdg.graph_name = self.graph_name
		return hdg


if __name__ == '__main__':
	loader_config.source_type = SourceType.LOCAL_FILE
	loader_config.source_uris = ['../data/ent_test1.csv', '../data/ent_test2.csv', '../data/rel_test.csv']
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

