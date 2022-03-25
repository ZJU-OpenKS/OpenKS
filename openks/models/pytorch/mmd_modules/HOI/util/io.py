import os
import pickle
import json
import yaml
import numpy as np
import gzip
import scipy.io


def load_pickle_object(file_name, compress=True):
    data = read(file_name)
    if compress:
        load_object = pickle.loads(gzip.decompress(data))
    else:
        load_object = pickle.loads(data)
    return load_object


def dump_pickle_object(dump_object, file_name, compress=True, compress_level=9):
    data = pickle.dumps(dump_object)
    if compress:
        write(file_name, gzip.compress(data, compresslevel=compress_level))
    else:
        write(file_name, data)


def load_json_object(file_name, compress=False):
    if compress:
        return json.loads(gzip.decompress(read(file_name)).decode('utf8'))
    else:
        return json.loads(read(file_name, 'r'))


def dump_json_object(dump_object, file_name, compress=False, indent=4):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode('utf8')))
    else:
        write(file_name, data, 'w')


def dumps_json_object(dump_object, indent=4):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
    return data


def load_mat_object(file_name):
    return scipy.io.loadmat(file_name=file_name)


def load_yaml_object(file_name):
    return yaml.load(read(file_name, 'r'))


def read(file_name, mode='rb'):
    with open(file_name, mode) as f:
        return f.read()


def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)


def serialize_object(in_obj, method='json'):
    if method == 'json':
        return json.dumps(in_obj)
    else:
        return pickle.dumps(in_obj)


def deserialize_object(obj_str, method='json'):
    if method == 'json':
        return json.loads(obj_str)
    else:
        return pickle.loads(obj_str)


def mkdir_if_not_exists(dir_name, recursive=False):
    if os.path.exists(dir_name):
        return

    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class JsonSerializableClass():
    def to_json(self, json_filename=None):
        serialized_dict = json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)
        serialized_dict = json.loads(serialized_dict)
        if json_filename is not None:
            dump_json_object(serialized_dict, json_filename)

        return serialized_dict

    def from_json(self, json_filename):
        assert (type(json_filename is dict)), 'Use from dict instead'
        dict_to_restore = load_json_object(json_filename)
        for attr_name, attr_value in dict_to_restore.items():
            setattr(self, attr_name, attr_value)

    def from_dict(self, dict_to_restore):
        for attr_name, attr_value in dict_to_restore.items():
            setattr(self, attr_name, attr_value)


class WritableToFile():
    def to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(self.__str__())