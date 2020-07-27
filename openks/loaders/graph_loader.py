"""
Loader for generating knowledge graph data format MTG
"""
import os
import json
from zipfile import ZipFile
import logging
from py2neo import Graph,Node
from .loader import Loader, LoaderConfig, SourceType, FileType
from ..abstract.mtg import MTG
import pdb

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
		self.graph = self._load_data()
		self.graph.name = graph_name if graph_name else config.data_name

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
		file_names = []
		entities = {}
		relations = {}
		global_count = 0
		if self.config.file_type == FileType.CSV:
			if self.config.source_uris.endswith('.zip'):
				with ZipFile(self.config.source_uris) as zf:
					file_names = [os.path.splitext(os.path.basename(item))[0] for item in zf.namelist() if item.endswith('.csv')]
			elif self.config.source_uris.endswith('.csv'):
				file_names = [os.path.splitext(os.path.basename(path))[0] for path in self.config.source_uris]
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
							'axiom': ((name.split('_')[1], name.split('_')[2]),)
						}
					)
					rel_index.append(count)
				else:
					logger.warn("File name {} is not allowed for graph loader, should start with 'ent_' or 'rel_'".format(name))
					return None
				count += 1
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

		elif self.config.file_type == FileType.CNSCHEMA:
			raise NotImplementedError

		elif self.config.file_type == FileType.OPENBASE:
			# schema from OpenBase
			headers = self.dataset.headers[0]
			bodies = self.dataset.bodies[0]
			# entity concepts
			ent_type_col = [head for head in headers if head.startswith('@type_')]
			ent_type_col.sort()
			ent_type_index = [headers.index(col) for col in ent_type_col]
			for body in bodies:
				for i in range(len(ent_type_index)):
					if i+1 < len(ent_type_index):
						parent = 'openks/'+ self.graph_name +'/class/' + body[ent_type_index[i+1]]
					else:
						parent = None
					concept = {
						'id': 'openks/'+ self.graph_name +'/class/' + body[ent_type_index[i]], 
						'name': body[ent_type_index[i]], 
						'type': 'entity', 
						'parent': parent
					}
					if concept not in schema['concepts']:
						schema['concepts'].append(concept)
			# relation concepts
			rel_types = ['_'.join(head.split('_')[:-2]) for head in headers if head.endswith('_@refer')]
			rel_types = list(set(rel_types))
			for rel_type in rel_types:
				from_type = ''
				to_type = ''
				rel_index = headers.index(rel_type + '_0_@refer')
				id_index = headers.index('@id')
				type_index = headers.index('@type_0')
				for body in bodies:
					if body[rel_index]:
						from_type = 'openks/'+ self.graph_name +'/class/' + body[type_index]
						for item in bodies:
							if item[id_index] == body[rel_index]:
								to_type = 'openks/'+ self.graph_name +'/class/' + item[type_index]
								break
						break
				concept = {
					'id': 'http://openbase/'+ self.graph_name +'/class/' + rel_type, 
					'name': rel_type, 
					'type': 'relation', 
					'axiom': ((from_type, to_type),)
				}
				if concept not in schema['concepts']:
					schema['concepts'].append(concept)
			# attributes 
			tmp = [head.split('_@refer')[0] for head in headers if head.endswith('_@refer')]
			for head in headers:
				flag = True
				for item in tmp:
					if head.startswith(item):
						attr_id = '_'.join(head.split('_')[:-2]) + '_' + head.split('_')[-1]
						attribute = {
							'id': 'openks/attribute/' + attr_id, 
							'name': attr_id, 
							'type': 'relation', 
							'concept': 'http://openbase/'+ self.graph_name +'/class/' + '_'.join(item.split('_')[:-1]),
							'value': 'any'
						}
						flag = False
						if attribute not in schema['attributes']:
							schema['attributes'].append(attribute)
						break
					else:
						continue
				
				if flag:
					attribute = {
						'id': 'openks/attribute/' + head, 
						'name': head, 
						'type': 'entity', 
						'concept': None,
						'value': 'any'
					}
					if attribute not in schema['attributes']:
						schema['attributes'].append(attribute)


			# entity from OpenBase
			ent_index = [headers.index(ent_attr) for ent_attr in [item['name'] for item in schema['attributes'] if item['type'] == 'entity']]
			ent_tmp = {}
			count = 0
			for index in ent_index:
				ent_tmp[headers[index]] = count
				count += 1
			for body in bodies:
				if body[ent_type_index[0]] in [item['name'] for item in schema['concepts']]:
					if body[ent_type_index[0]] not in entities:
						entities[body[ent_type_index[0]]] = {'pointer': ent_tmp, 'instances': []}
					row = []
					for index in ent_index:
						row.append(body[index])
					entities[body[ent_type_index[0]]]['instances'].append(tuple(row))

			# relation from OpenBase
			rel_tmp = {}
			for head in headers:
				for item in rel_types:
					if head.startswith(item):
						if item not in rel_tmp:
							rel_tmp[item] = {}
						for rel_attr in schema['attributes']:
							if rel_attr['concept'] == 'http://openbase/'+ self.graph_name +'/class/' + item:
								if rel_attr['name'].split('_')[-1] == head.split('_')[-1]:
									if rel_attr['name'] not in rel_tmp[item]:
										rel_tmp[item][rel_attr['name']] = []
									rel_tmp[item][rel_attr['name']].append(headers.index(head))
			for body in bodies:
				for rel_type in rel_tmp.keys():
					if rel_type not in relations:
						relations[rel_type] = {'pointer': {'appID': 0}, 'instances': []}
					for i in range(len(list(rel_tmp[rel_type].values())[0])):
						rel_item = [body[headers.index('appID')]]
						count = 1
						for k in rel_tmp[rel_type].keys():
							relations[rel_type]['pointer'][k] = count
							count += 1
							rel_item.append(body[rel_tmp[rel_type][k][i]])
						for item in rel_item[1:]:
							if item:
								relations[rel_type]['instances'].append(tuple(rel_item))
								break

		elif self.config.file_type == FileType.OPENKS:
			schema = []
			entities = []
			relations = []
			if os.path.exists(self.config.source_uris + '/schema.json'):
				with open(self.config.source_uris + '/schema.json', 'r') as f:
					schema = json.load(f)
				for entity in self.dataset.bodies[0]:
					if len(entity) == 2:
						entities.append((int(entity[0]), entity[1], tuple([])))
					else:
						entities.append((int(entity[0]), entity[1], tuple(entity[2:])))
				for relation in self.dataset.bodies[1]:
					if len(relation) == 3:
						relations.append(((int(relation[0]), relation[1], int(relation[2])), tuple([])))
					else:
						relations.append(((int(relation[0]), relation[1], int(relation[2])), tuple(relation[3:])))

			else:
				logger.warn("A schema JSON file must exists!")
				raise IOError

		mtg.schema = schema
		mtg.entities = entities
		mtg.triples = relations
		return mtg

	def graph2neo(self, graph: MTG, graph_db, clean=True):
		create = True
		if len(graph_db.nodes) > 0:
			if clean:
				graph_db.delete_all()
				logger.info("Cleaned all nodes and relations in graph.")
			else:
				create = False
				logger.info("Graph contains nodes and edges and no new nodes will be imported.")
		if create:
			# create neo4j nodes
			count = 0
			for item in graph.entities:
				props = [struct['properties'] for struct in graph.schema if struct['type'] == 'entity' and struct['concept'] == item[1]]
				prop_names = [prop['name'] for prop in props[0]]
				prop_names.append('gid')
				prop_values = list(item[2])
				prop_values.append(item[0])
				prop_dict = dict(zip(prop_names, prop_values))
				node = Node(item[1], **prop_dict)
				graph_db.create(node)
				count += 1
				if count % 1000 == 0:
					logger.info("Already imported nodes %d, total nodes %d" % (count, len(graph.entities)))
			# create neo4j edges
			start_node = ''
			end_node = ''
			count = 0
			for item in graph.triples:
				p = item[0][0]
				q = item[0][2]
				rel_type = item[0][1]
				rel_name = item[1][0] if item[1] else rel_type
				for struct in graph.schema:
					if struct['concept'] == item[0][1] and struct['type'] == 'relation':
						start_node = struct['members'][0]
						end_node = struct['members'][1]
				query = "match(p:%s),(q:%s) where p.gid=%d and q.gid=%d create (p)-[rel:%s{name:'%s'}]->(q)" % (start_node, end_node, p, q, rel_type, rel_name)
				try:
					graph_db.run(query)
					count += 1
					if count % 1000 == 0:
						logger.info("Already imported edges %d, total edges %d, current relation type %s" % (count, len(graph.triples), rel_type))
				except Exception as e:
					print(e)
		return None