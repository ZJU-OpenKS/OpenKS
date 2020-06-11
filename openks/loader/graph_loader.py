"""
Loader for generating knowledge graph data format HDG
"""
import os
import logging
from loader import Loader, LoaderConfig, SourceType
import sys
sys.path.append('..')
from abstract.hdg import HDG

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class GraphLoader(Loader):

	def __init__(self, config: LoaderConfig) -> None:
		super(GraphLoader, self).__init__(config)
		self.graph = self._load_data()

	def _load_data(self) -> HDG:
		""" 
		transform MDD from _read_data method to HDG. 
		We require an entity dataset has the file name likes 'ent_<entity_type>', each line in the file has at least an 'id' column. 
		We require a relation dataset has the file name likes 'rel_<relation_type>', each line in the file has at least two id columns for head entities and tail entities """
		self.config
		self.dataset
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
				logging.warn("File name " + name + "is not allowed for graph loader, should start with 'ent_' or 'rel_'")
				return None
			count += 1
		HDG.entity_types = entity_types
		HDG.relation_types = relation_types
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
		HDG.entity_attrs = entity_attrs
		HDG.relation_attrs = relation_attrs
		HDG.entities = entities
		HDG.relations = relations


if __name__ == '__main__':
	LoaderConfig.source_type = SourceType.LOCAL_FILE
	LoaderConfig.source_uris = ['../data/ent_test1.csv', '../data/ent_test2.csv', '../data/rel_test.csv']
	LoaderConfig.data_name = 'test_graph'
	graph_loader = GraphLoader(LoaderConfig)
	print(HDG.entity_types)
	print(HDG.relation_types)
	print(HDG.entity_attrs)
	print(HDG.relation_attrs)
	for line in HDG.entities:
		print(line)
	for line in HDG.relations:
		print(line)

