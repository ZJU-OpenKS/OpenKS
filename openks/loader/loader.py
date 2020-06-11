"""
Basic class for loading data from file or database systems
"""
import sys
sys.path.append('..')
from enum import Enum, unique
import csv
import logging
from abstract.mdd import MDD

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

@unique
class SourceType(Enum):
	LOCAL_FILE = 'local_file'
	HDFS = 'hdfs'


class LoaderConfig(object):

	def __init__(self):
		self._source_type = SourceType.LOCAL_FILE
		self._source_uri = ''
		self._data_name = ''

	@property
	def source_type(self):
		return self._source_type
	
	@source_type.setter
	def source_type(self, source_type: Enum):
		self._source_type = source_type

	@property
	def source_uri(self):
		return self._source_uri
	
	@source_uri.setter
	def source_uri(self, source_uri: str):
		self._source_uri = source_uri

	@property
	def data_name(self):
		return self._data_name
	
	@data_name.setter
	def data_name(self, data_name: str):
		self._data_name = data_name


class Loader(object):
	""" basic loader from multiple data sources """

	def __init__(self, config: LoaderConfig):
		self.config = config
		self.dataset = self._read_data()

	def _read_data(self) -> MDD:
		""" read data from multiple sources and return MDD """
		if self.config.source_type == SourceType.LOCAL_FILE:
			return self._read_file()
		elif self.config.source_type == SourceType.HDFS:
			return self._read_hdfs()
		else:
			logging.warn("The source type " + LoaderConfig.source_type + "has not been implemented yet.")
			return NotImplemented


	def _read_file(self):
		""" Currently support csv file format from local """
		csv_reader = csv.reader(open(self.config.source_uri, encoding='utf-8'))
		MDD.headers = next(csv_reader)
		MDD.body = csv_reader
		MDD.name = self.config.data_name
		return MDD

	def _read_hdfs(self):
		""" Access HDFS with delimiter """
		return NotImplemented

if __name__ == '__main__':
	LoaderConfig.source_type = SourceType.LOCAL_FILE
	LoaderConfig.source_uri = '../data/test.csv'
	LoaderConfig.data_name = 'test'
	loader = Loader(LoaderConfig)
	print(loader.dataset.headers)
	for line in loader.dataset.body:
		print(line)