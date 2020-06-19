"""
Loader for generating knowledge graph data format MTG
"""
import os
from zipfile import ZipFile
import logging
from .loader import Loader, LoaderConfig, SourceType
from ..abstract.mtg import MTG

logger = logging.getLogger(__name__)

loader_config = LoaderConfig()
mtg = MTG()

class GraphLoader(Loader):
	"""
	Specific loader for generating MTG format from MMD
	A knowledge graph structure should follows:
		entity_types: list<str>
		relation_types: list<str>
		entity_attrs: dict<str, list<str>>
		relation_attrs: dict<str, dict<str, dict<str, str>/list<str>>> // must have 'from', 'to', 'attrs' keys
		entities: dict<str, list<list<T>>>
		relations: dict<str, list<list<T>>>
	"""
	def __init__(
		self, 
		config: LoaderConfig, 
		graph_name: str = ''
		) -> None:
		super(GraphLoader, self).__init__(config)
		self.graph_name = config.data_name
		self.graph = self._load_data()

	def _load_data(self) -> MTG:
		""" 
		transform MMD from _read_data method to MTG. 
		We require an entity dataset has the file name likes 'ent_<entity_type>', each line in the file has at least an 'id' column. 
		We require a relation dataset has the file name likes 'rel_<relation_type>', each line in the file has at least two id columns for head entities and tail entities """
		entity_types = []
		relation_types = []
		ent_index = []
		rel_index = []
		count = 0
		file_names = None
		if isinstance(self.config.source_uris, str) and self.config.source_uris.endswith('.zip'):
			with ZipFile(self.config.source_uris) as zf:
				file_names = [os.path.splitext(os.path.basename(item))[0] for item in zf.namelist() if item.endswith('.csv')]
		elif isinstance(self.config.source_uris, list):
			file_names = [os.path.splitext(os.path.basename(path))[0] for path in self.config.source_uris]
		for name in file_names:
			if name.startswith('ent_'):
				entity_types.append('_'.join(name.split('_')[1:]))
				ent_index.append(count)
			elif name.startswith('rel_'):
				relation_types.append('_'.join(name.split('_')[1:]))
				rel_index.append(count)
			else:
				logger.warn("File name {} is not allowed for graph loader, should start with 'ent_' or 'rel_'".format(name))
				return None
			count += 1
		mtg.entity_types = entity_types
		mtg.relation_types = relation_types
		entity_attrs = {}
		relation_attrs = {}
		entities = {}
		relations = {}
		for ent_type, index in zip(entity_types, ent_index):
			entity_attrs[ent_type] = self.dataset.headers[index]
			entities[ent_type] = self.dataset.bodies[index]
		for rel_type, index in zip(relation_types, rel_index):
			tmp = self.config.ent_rel_mapping[rel_type]
			tmp.update({'attrs': self.dataset.headers[index]})
			relation_attrs[rel_type] = tmp
			relations[rel_type] = self.dataset.bodies[index]
		mtg.entity_attrs = entity_attrs
		mtg.relation_attrs = relation_attrs
		mtg.entities = entities
		mtg.relations = relations
		mtg.graph_name = self.graph_name
		return mtg


if __name__ == '__main__':
	loader_config.source_type = SourceType.LOCAL_FILE
	loader_config.source_uris = ['openks/data/ent_test1.csv', 'openks/data/ent_test2.csv', 'openks/data/rel_test.csv']
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

