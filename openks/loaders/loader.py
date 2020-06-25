"""
Basic class for loading data from file or database systems
"""
from typing import List
from enum import Enum, unique
import csv
from zipfile import ZipFile
from io import TextIOWrapper
import json
import logging
from ..abstract.mmd import MMD

logger = logging.getLogger(__name__)

@unique
class SourceType(Enum):
	LOCAL_FILE = 'local_file'
	HDFS = 'hdfs'

@unique
class FileType(Enum):
	CSV = 'csv'
	OPENKG = 'openkg'
	OPENBASE = 'openbase'


def flatten_json(y):
	out = {}
	def flatten(x, name=''):
		if type(x) is dict:
			for a in x:
				flatten(x[a], name + a + '_')
		elif type(x) is list:
			i = 0
			for a in x:
				flatten(a, name + str(i) + '_')
				i += 1
		else:
			out[name[:-1]] = x
	flatten(y)
	return out


class LoaderConfig(object):
	"""
	The config object to load data from various sources
	"""
	def __init__(
		self, 
		source_type: SourceType = SourceType.LOCAL_FILE, 
		file_type: FileType = FileType.CSV,
		source_uris: List = [], 
		data_name: str = ''
		) -> None:
		self._source_type = source_type
		self._file_type = file_type
		# support loading multiple files
		self._source_uri = source_uris
		self._data_name = data_name

	@property
	def source_type(self):
		return self._source_type
	
	@source_type.setter
	def source_type(self, source_type: Enum):
		self._source_type = source_type

	@property
	def file_type(self):
		return self._file_type
	
	@file_type.setter
	def file_type(self, file_type: Enum):
		self._file_type = file_type

	@property
	def source_uris(self):
		return self._source_uris
	
	@source_uris.setter
	def source_uris(self, source_uris: str):
		self._source_uris = source_uris

	@property
	def data_name(self):
		return self._data_name
	
	@data_name.setter
	def data_name(self, data_name: str):
		self._data_name = data_name


loader_config = LoaderConfig()
mmd = MMD()

class Loader(object):
	""" basic loader from multiple data sources """

	def __init__(self, config: loader_config) -> None:
		self.config = config
		self.dataset = self._read_data()

	def _read_data(self) -> MMD:
		""" read data from multiple sources and return MMD """
		if self.config.source_type == SourceType.LOCAL_FILE:
			return self._read_files()
		elif self.config.source_type == SourceType.HDFS:
			return self._read_hdfs()
		else:
			raise NotImplementedError("The source type {} has not been implemented yet.".format(loader_config.source_type))


	def _read_files(self) -> MMD:
		""" Currently support csv file format from local
		    support *.csv and *labels.csv files either in a zip file or directly in a folder """
		headers = []
		bodies = []
		if self.config.file_type == FileType.CSV:
			if self.config.source_uris.endswith('.zip'):
				with ZipFile(self.config.source_uris) as zf:
					for item in zf.namelist():
						if item.endswith('.csv'):
							# with zf.open(item, 'r') as infile:
							csv_reader = csv.reader(TextIOWrapper(zf.open(item, 'r'), 'utf-8'))
							headers.append(next(csv_reader))
							# need to find a more efficient way, the csv reader is a generator that can only be used once
							bodies.append(list(csv_reader))
			elif self.config.source_uris.endswith('.csv'):
				for uri in self.config.source_uris:
					if uri.endswith('.csv'):
						csv_reader = csv.reader(open(uri, newline='', encoding='utf-8'))
						headers.append(next(csv_reader))
						bodies.append(list(csv_reader))
		elif self.config.file_type == FileType.OPENKG:
			header = ['@id', 'label_@language', 'label_@value']
			body = []
			with open(self.config.source_uris, 'r') as load_f:
				load_dict = json.load(load_f)
				header.extend(load_dict['@context'].keys())
				header = [h for h in header if h not in ['label', 'range', 'domain', 'subClassOf']]
				tmp_h = [h for h in header if h not in ['@id', '@language', '@value']]
				for item in load_dict['@graph']:
					if item['@id'].split('/')[-2] == 'resource':
						row = [item['@id'], item['label']['@language'], item['label']['@value']]
						for h in tmp_h:
							if h in item:
								row.append(item[h])
							else:
								row.append(None)
						body.append(tuple(row))
			headers.append(tuple(header))
			bodies.append(body)
		elif self.config.file_type == FileType.OPENBASE:
			header = []
			body = []
			with open(self.config.source_uris, 'r') as load_f:
				for line in load_f:
					row = []
					flat_line = flatten_json(json.loads(line))
					for key in flat_line:
						if key not in header:
							header.append(key)
					for h in header:
						if h in flat_line:
							row.append(flat_line[h])
						else:
							row.append(None)
					body.append(row)
			for item in body:
				if len(item) < len(header):
					item.extend([None for i in range(len(header) - len(item))])
			headers.append(tuple(header))
			bodies.append(tuple([tuple(item) for item in body]))

		mmd.name = self.config.data_name
		mmd.headers = headers
		mmd.bodies = bodies
		return mmd

	def _read_hdfs(self):
		""" Access HDFS with delimiter """
		raise NotImplementedError()


if __name__ == '__main__':
	loader_config.source_type = SourceType.LOCAL_FILE
	loader_config.source_uris = ['openks/data/ent_test1.csv', 'openks/data/ent_test2.csv', 'openks/data/rel_test.csv']
	loader_config.data_name = 'test'
	loader = Loader(loader_config)
	print(mmd.headers)
	for body in mmd.bodies:
		for line in body:
			print(line)
