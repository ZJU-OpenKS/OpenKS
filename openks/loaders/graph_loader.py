"""
Loader for generating knowledge graph data format MTG
"""
import os
from zipfile import ZipFile
import logging
from .loader import Loader, LoaderConfig, SourceType, FileType
from ..abstract.mtg import MTG

logger = logging.getLogger(__name__)

loader_config = LoaderConfig()
mtg = MTG()

class GraphLoader(Loader):
	"""
	Specific loader for generating MTG format from MMD
	"""
	def __init__(
		self, 
		config: LoaderConfig, 
		graph_name: str = ''
		) -> None:
		super(GraphLoader, self).__init__(config)
		self.graph_name = graph_name if graph_name else config.data_name
		self.graph = self._load_data()

	def _load_data(self) -> MTG:
		""" 
		transform MMD from _read_data method to MTG. 
		We require an entity dataset has the file name likes 'ent_<entity_type>', each line in the file has at least an 'id' column. 
		We require a relation dataset has the file name likes 'rel_<from_entity>_<to_entity>', 
			each line in the file has at least two id columns for head entities and tail entities with format like  <entity_name>_id"""
		schema = {'concepts': [], 'attributes': []}
		ent_index = []
		rel_index = []
		count = 0
		file_names = None
		if self.config.file_type == FileType.ZIP:
			with ZipFile(self.config.source_uris) as zf:
				file_names = [os.path.splitext(os.path.basename(item))[0] for item in zf.namelist() if item.endswith('.csv')]
		elif self.config.file_type == FileType.CSV:
			file_names = [os.path.splitext(os.path.basename(path))[0] for path in self.config.source_uris]
		elif self.config.file_type == FileType.JSON:
			raise NotImplementedError
		for name in file_names:
			if name.startswith('ent_'):
				schema['concepts'].append(
					{
						'id': 'openks/concept/' + str(count), 
						'name': '_'.join(name.split('_')[1:]), 
						'type': 'entity', 
						'parent': None
					}
				)
				ent_index.append(count)
			elif name.startswith('rel_'):
				schema['concepts'].append(
					{
						'id': 'openks/concept/' + str(count), 
						'name': '_'.join(name.split('_')[1:]), 
						'type': 'relation',
						'from': name.split('_')[1],
						'to': name.split('_')[2]
					}
				)
				rel_index.append(count)
			else:
				logger.warn("File name {} is not allowed for graph loader, should start with 'ent_' or 'rel_'".format(name))
				return None
			count += 1
		entities = {}
		relations = {}
		global_count = 0
		for index, concept in enumerate(schema['concepts']):
			tmp = {}
			count = 0
			for attr in self.dataset.headers[index]:
				schema['attributes'].append(
					{
						'id': 'openks/attribute/' + str(global_count), 
						'name': attr, 
						'type': concept['type'], 
						'concept': concept['id'], 
						'value': 'any'
					}
				)
				tmp[attr] = count
				count += 1
				global_count += 1

			if index in ent_index:
				entities[concept['name']] = {
					'pointer': tmp, 
					'instances': self.dataset.bodies[index]
				}
			else:
				relations[concept['name']] = {
					'pointer': tmp, 
					'instances': self.dataset.bodies[index]
				}

		mtg.schema = schema
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

