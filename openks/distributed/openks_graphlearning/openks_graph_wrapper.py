# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as L

from pgl.utils import op
from pgl.utils import paddle_helper
from pgl.utils.logger import log
import pgl.graph_wrapper as pg
import pgl.heter_graph_wrapper as hpg

ALL = "__ALL__"
__all__ = ["GraphWrapper", "StaticGraphWrapper", "BatchGraphWrapper"]

class GraphModule(object):
    """ Interface for graph layers.
    """
    def __init__(self, **kwargs):
        raise NotImplemented()

    def __call__(self, gw, **kwargs):
        feat = self.propagate(**kwargs)
        message = gw.send(self.send, nfeat_list=feat)
        output = gw.recv(recv_func, message)
        return output

    def send(self, src_feat, dst_feat, edge_feat):
        raise NotImplemented()

    def recv(self, feat):
        raise NotImplemented()

    def propagate(self, **kwargs):
        raise NotImplemented()


class GraphWrapper(object):
    def __init__(self, name, node_feat=[], edge_feat=[], **kwargs):
        self._gw = pg.GraphWrapper(name, node_feat, edge_feat, **kwargs)
    def __repr__(self):
        return self._data_name_prefix

    def send(self, message_func, nfeat_list=None, efeat_list=None):
        return self._gw.send(message_func, nfeat_list, efeat_list)

    def recv(self, msg, reduce_function):
        return self._gw.recv(msg, reduce_function)

    @property
    def edges(self):
        return self._gw.edges

    @property
    def num_nodes(self):
        return self._gw._num_nodes

    @property
    def graph_lod(self):
        return self._gw._graph_lod

    @property
    def num_graph(self):
        return self._gw._num_graph

    @property
    def edge_feat(self):
        return self._gw.edge_feat_tensor_dict

    @property
    def node_feat(self):
        return self._gw.node_feat_tensor_dict

    def indegree(self):
        return self._gw._indegree
    
    def to_feed(self, graph):
        return self._gw.to_feed(graph)

class StaticGraphWrapper(GraphWrapper):
    def __init__(self, name, graph, place):
        self._gw = pg.StaticGraphWrapper(name, graph, place)

class BatchGraphWrapper(GraphWrapper):
    def __init__(self, num_nodes, num_edges, edges, node_feats=None, edge_feats=None):
        self._gw = pg.BatchGraphWrapper(name, graph, place)

class HeterGraphWrapper(GraphWrapper):
    def __init__(self, name, edge_types, node_feat={}, edge_feat={}, **kwargs):
        self._gw = hpg.HeterGraphWrapper(name, edge_types, node_feat, edge_feat, **kwargs)

    def to_feed(self, heterGraph, edge_types_list=ALL):
        return self._gw.to_feed(heterGraph, edge_types_list)
