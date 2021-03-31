

# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    This package implement Graph structure for handling graph data.
"""

import os
import numpy as np
import pickle as pkl
import time
from collections import defaultdict
import abc
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as L

from pgl.utils import op
from pgl.utils import paddle_helper
from pgl.utils.logger import log
import pgl.graph  as pg
import pgl.heter_graph as hpg

class graph_base(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def dump(self, path):
        """Save node information and edge information.
        """
        pass

    @abc.abstractmethod
    def edge_feat(self):
        """Return edge features.
        """
        pass

    @abc.abstractmethod
    def node_feat(self):
        """Return node features.
        """
        pass

    @abc.abstractmethod
    def num_edges(self):
        """Return edges number.
        """
        pass

    @abc.abstractmethod
    def num_nodes(self):
        """Return edges number.
        """
        pass

    @abc.abstractmethod
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        pass

    @abc.abstractmethod
    def indegree(self, nodes):
        """Return the indegree of the given nodes as numpy.ndarray.
        """
        pass

    @abc.abstractmethod
    def outdegree(self, nodes):
        """Return the outdegree of the given nodes as numpy.ndarray.
        """
        pass

    @abc.abstractmethod
    def successor(self, nodes, 
                  return_eids):
        """ Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of successor ids for given nodes.
        """
        pass

    @abc.abstractmethod
    def sample_successor(self, nodes,
                         max_degree, 
                         return_eids, 
                         shuffle):
        """Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled successor ids for given nodes.
        """
        pass

    @abc.abstractmethod
    def predecessor(self, nodes, return_eids):
        """Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of predecessor ids for given nodes.
        """
        pass

    @abc.abstractmethod
    def sample_predecessor(self, nodes,
                           max_degree,
                           return_eids,
                           shuffle):
        """Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled predecessor ids for given nodes.
        """
        pass

    @abc.abstractmethod
    def node_feat_info(self):
        """Return the information of node feature for GraphWrapper as a list of tuple (name, shape, dtype) for all given node feature
        """
        pass

    @abc.abstractmethod
    def edge_feat_info(self):
        """Return the information of edge feature for GraphWrapper as a list of tuple (name, shape, dtype) for all given edge feature.
        """
        pass

    @abc.abstractmethod
    def node_batch_iter(self, batch_size, shuffle):
        """Node batch iterator.
        """
        pass

    @abc.abstractmethod
    def sample_nodes(self, sample_num):
        """Sample nodes from the graph, return a list of nodes.
        """
        pass

















class Graph(object):
    """Implementation of graph structure in pgl.
    This is a simple implementation of graph structure in pgl.
    Args:
        num_nodes: number of nodes in a graph
        edges: list of (u, v) tuples
        node_feat (optional): a dict of numpy array as node features
        edge_feat (optional): a dict of numpy array as edge features (should
                                have consistent order with edges)
    Examples:
        .. code-block:: python
            import numpy as np
            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            feature = np.random.randn(5, 100)
            edge_feature = np.random.randn(3, 100)
            graph = Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        },
                        edge_feat={
                            "edge_feature": edge_feature
                        })
    """

    def __init__(self, num_nodes, edges=None, node_feat=None, edge_feat=None):
        self._g = pg(num_nodes, edges, node_feat, edge_feat)

    def dump(self, path):
        self._g.dump(path)

    @property
    def adj_src_index(self):
        """Return an EdgeIndex object for src.
        """

        return self._g.adj_src_index()

    @property
    def adj_dst_index(self):
        """Return an EdgeIndex object for dst.
        """
        return self._g.adj_dst_index()

    @property
    def edge_feat(self):
        """Return a dictionary of edge features.
        """
        return self._g.edge_feat()

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        return self._g.node_feat()

    @property
    def num_edges(self):
        """Return the number of edges.
        """
        return self._g.num_edges()

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._g.num_nodes()

    @property
    def edges(self):
        """Return all edges in numpy.ndarray with shape (num_edges, 2).
        """
        return self._g.edges()

    def sorted_edges(self, sort_by="src"):
        """Return sorted edges with different strategies.
        This function will return sorted edges with different strategy.
        If :code:`sort_by="src"`, then edges will be sorted by :code:`src`
        nodes and otherwise :code:`dst`.
        Args:
            sort_by: The type for sorted edges. ("src" or "dst")
        Return:
            A tuple of (sorted_src, sorted_dst, sorted_eid).
        """
        return self._g.sorted_edges(sort_by)

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        return self._g.nodes()

    def indegree(self, nodes=None):
        """Return the indegree of the given nodes
        This function will return indegree of given nodes.
        Args:
            nodes: Return the indegree of given nodes,
                   if nodes is None, return indegree for all nodes
        Return:
            A numpy.ndarray as the given nodes' indegree.
        """
        return self._g.indegree(nodes)

    def outdegree(self, nodes=None):
        """Return the outdegree of the given nodes.
        This function will return outdegree of given nodes.
        Args:
            nodes: Return the outdegree of given nodes,
                   if nodes is None, return outdegree for all nodes
        Return:
            A numpy.array as the given nodes' outdegree.
        """
        return self._g.outdegree(nodes)

    def successor(self, nodes=None, return_eids=False):
        """Find successor of given nodes.
        This function will return the successor of given nodes.
        Args:
            nodes: Return the successor of given nodes,
                   if nodes is None, return successor for all nodes.
            return_eids: If True return nodes together with corresponding eid
        Return:
            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of successor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their successors.
        Example:
            .. code-block:: python
                import numpy as np
                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                graph = Graph(num_nodes=num_nodes,
                        edges=edges)
                succ, succ_eid = graph.successor(return_eids=True)
            This will give output.
            .. code-block:: python
                succ:
                      [[1],
                       [2],
                       [],
                       [4],
                       []]
                succ_eid:
                      [[0],
                       [1],
                       [],
                       [2],
                       []]
        """
        return self._g.successor(nodes, return_eids)

    def sample_successor(self,
                         nodes,
                         max_degree,
                         return_eids=False,
                         shuffle=False):
        """Sample successors of given nodes.
        Args:
            nodes: Given nodes whose successors will be sampled.
            max_degree: The max sampled successors for each nodes.
            return_eids: Whether to return the corresponding eids.
        Return:
            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled successor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their successors.
        """

        return self._g.sample_successor(
                         nodes,
                         max_degree,
                         return_eids,
                         shuffle)

    def predecessor(self, nodes=None, return_eids=False):
        """Find predecessor of given nodes.
        This function will return the predecessor of given nodes.
        Args:
            nodes: Return the predecessor of given nodes,
                   if nodes is None, return predecessor for all nodes.
            return_eids: If True return nodes together with corresponding eid
        Return:
            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of predecessor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their predecessors.
        Example:
            .. code-block:: python
                import numpy as np
                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                graph = Graph(num_nodes=num_nodes,
                        edges=edges)
                pred, pred_eid = graph.predecessor(return_eids=True)
            This will give output.
            .. code-block:: python
                pred:
                      [[],
                       [0],
                       [1],
                       [],
                       [3]]
                pred_eid:
                      [[],
                       [0],
                       [1],
                       [],
                       [2]]
        """
        return self._g.predecessor(nodes, return_eids)

    def sample_predecessor(self,
                           nodes,
                           max_degree,
                           return_eids=False,
                           shuffle=False):
        """Sample predecessor of given nodes.
        Args:
            nodes: Given nodes whose predecessor will be sampled.
            max_degree: The max sampled predecessor for each nodes.
            return_eids: Whether to return the corresponding eids.
        Return:
            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled predecessor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their predecessors.
        """
        return self._g.sample_predecessor(nodes,max_degree,return_eids,shuffle)

    def node_feat_info(self):
        """Return the information of node feature for GraphWrapper.
        This function return the information of node features. And this
        function is used to help constructing GraphWrapper
        Return:
            A list of tuple (name, shape, dtype) for all given node feature.
        Examples:
            .. code-block:: python
                import numpy as np
                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                feature = np.random.randn(5, 100)
                graph = Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        })
                print(graph.node_feat_info())
            The output will be:
            .. code-block:: python
                [("feature", [None, 100], "float32")]
        """

        return self._g.node_feat_info()

    def edge_feat_info(self):
        """Return the information of edge feature for GraphWrapper.
        This function return the information of edge features. And this
        function is used to help constructing GraphWrapper
        Return:
            A list of tuple (name, shape, dtype) for all given edge feature.
        Examples:
            .. code-block:: python
                import numpy as np
                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                feature = np.random.randn(3, 100)
                graph = Graph(num_nodes=num_nodes,
                        edges=edges,
                        edge_feat={
                            "feature": feature
                        })
                print(graph.edge_feat_info())
            The output will be:
            .. code-block:: python
                [("feature", [None, 100], "float32")]
        """
        return self._g.edge_feat_info()

    def subgraph(self,
                 nodes,
                 eid=None,
                 edges=None,
                 edge_feats=None,
                 with_node_feat=True,
                 with_edge_feat=True):
        """Generate subgraph with nodes and edge ids.
        This function will generate a :code:`pgl.graph.Subgraph` object and
        copy all corresponding node and edge features. Nodes and edges will
        be reindex from 0. Eid and edges can't both be None.
        WARNING: ALL NODES IN EID MUST BE INCLUDED BY NODES
        Args:
            nodes: Node ids which will be included in the subgraph.
            eid (optional): Edge ids which will be included in the subgraph.
            edges (optional): Edge(src, dst) list which will be included in the subgraph.
    
            with_node_feat: Whether to inherit node features from parent graph.
            with_edge_feat: Whether to inherit edge features from parent graph.
        Return:
            A :code:`pgl.graph.Subgraph` object.
        """
        return self._g.subgraph(nodes,eid,edges,edge_feats,with_node_feat,with_edge_feat)

    def node_batch_iter(self, batch_size, shuffle=True):
        """Node batch iterator
        Iterate all node by batch.
        Args:
            batch_size: The batch size of each batch of nodes.
            shuffle: Whether shuffle the nodes.
        Return:
            Batch iterator
        """
        return self._g.node_batch_iter( batch_size, shuffle )

    def sample_nodes(self, sample_num):
        """Sample nodes from the graph
        This function helps to sample nodes from all nodes.
        Nodes might be duplicated.
        Args:
            sample_num: The number of samples
        Return:
            A list of nodes
        """
        return self._g.sample_nodes(sample_num)

    def sample_edges(self, sample_num, replace=False):
        """Sample edges from the graph
        This function helps to sample edges from all edges.
        Args:
            sample_num: The number of samples
            replace: boolean, Whether the sample is with or without replacement.
        Return:
            (u, v), eid 
            each is a numy.array with the same shape.
        """

        return self._g.sample_edges(sample_num, replace)

    def has_edges_between(self, u, v):
        """Check whether some edges is in graph.
        Args:
            u: a numpy.array of src nodes ID.
            v: a numpy.array of dst nodes ID.
        Return:
            exists: A numpy.array of bool, with the same shape with `u` and `v`,
                exists[i] is True if (u[i], v[i]) is a edge in graph, Flase otherwise.
        """
        return self._g.has_edges_between( u, v)

    def random_walk(self, nodes, max_depth):
        """Implement of random walk.
        This function get random walks path for given nodes and depth.
        Args:
            nodes: Walk starting from nodes
            max_depth: Max walking depth
        Return:
            A list of walks.
        """
        return self._g.random_walk( nodes, max_depth)

    def node2vec_random_walk(self, nodes, max_depth, p=1.0, q=1.0):
        """Implement of node2vec stype random walk.
        Reference paper: https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf.
        Args:
            nodes: Walk starting from nodes
            max_depth: Max walking depth
            p: Return parameter
            q: In-out parameter
        Return:
            A list of walks.
        """
        self._g.node2vec_random_walk( nodes, max_depth, p=1.0, q=1.0)

    @property
    def num_graph(self):
        """ Return Number of Graphs"""
        return self._g.num_graph()

    @property
    def graph_lod(self):
        """ Return Graph Lod Index for Paddle Computation"""
        return  self._g.graph_lod()




class HeterGraph(object):
    """Implementation of heterogeneous graph structure in pgl
    This is a simple implementation of heterogeneous graph structure in pgl.
    Args:
        num_nodes: number of nodes in a heterogeneous graph
        edges: dict, every element in dict is a list of (u, v) tuples.
        node_types (optional): list of (u, node_type) tuples to specify the node type of every node
        node_feat (optional): a dict of numpy array as node features
        edge_feat (optional): a dict of dict as edge features for every edge type
    Examples:
        .. code-block:: python
            import numpy as np
            num_nodes = 4
            node_types = [(0, 'user'), (1, 'item'), (2, 'item'), (3, 'user')]
            edges = {
                'edges_type1': [(0,1), (3,2)],
                'edges_type2': [(1,2), (3,1)],
            }
            node_feat = {'feature': np.random.randn(4, 16)}
            edges_feat = {
                'edges_type1': {'h': np.random.randn(2, 16)},
                'edges_type2': {'h': np.random.randn(2, 16)},
            }
            g = heter_graph.HeterGraph(
                            num_nodes=num_nodes,
                            edges=edges,
                            node_types=node_types,
                            node_feat=node_feat,
                            edge_feat=edges_feat)
    """

    def __init__(self,
                 num_nodes,
                 edges,
                 node_types=None,
                 node_feat=None,
                 edge_feat=None):

        self._g = hpg.HeterGraph(num_nodes,
                            edges,
                            node_types,
                            node_feat,
                            edge_feat)
        self._multi_graph = self._g._multi_graph

    def dump(self, path, indegree=False, outdegree=False):

        self._g.dump(path, indegree, outdegree)

    @property
    def edge_types(self):
        """Return a list of edge types.
        """
        return self._g.edge_types()

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._g.num_nodes()

    @property
    def num_edges(self):
        """Return edges number of all edge types.
        """

        return self._g.num_edges()

    @property
    def node_types(self):
        """Return the node types.
        """
        return self._g.node_types()

    @property
    def edge_feat(self, edge_type=None):
        """Return edge features of all edge types.
        """
        return self._g.edge_feat(edge_type)

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        return self._g.node_feat()

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        return self._g.nodes()

    def __getitem__(self, edge_type):
        """__getitem__
        """
        return self._g.__getitem__(edge_type)

    def num_nodes_by_type(self, n_type=None):
        """Return the number of nodes with the specified node type.
        """
        return self._g.num_nodes_by_type(n_type)

    def indegree(self, nodes=None, edge_type=None):
        """Return the indegree of the given nodes with the specified edge_type.
        Args:
            nodes: Return the indegree of given nodes.
                    if nodes is None, return indegree for all nodes.
            edge_types: Return the indegree with specified edge_type.
                    if edge_type is None, return the total indegree of the given nodes.
        Return:
            A numpy.ndarray as the given nodes' indegree.
        """


        return self._g.indegree(nodes, edge_type)

    def outdegree(self, nodes=None, edge_type=None):
        """Return the outdegree of the given nodes with the specified edge_type.
        Args:
            nodes: Return the outdegree of given nodes,
                   if nodes is None, return outdegree for all nodes
            edge_types: Return the outdegree with specified edge_type.
                    if edge_type is None, return the total outdegree of the given nodes.
        Return:
            A numpy.array as the given nodes' outdegree.
        """
        return self._g.outdegree(nodes, edge_type)

    def successor(self, edge_type, nodes=None, return_eids=False):
        """Find successor of given nodes with the specified edge_type.
        Args:
            nodes: Return the successor of given nodes,
                   if nodes is None, return successor for all nodes
            edge_types: Return the successor with specified edge_type.
                    if edge_type is None, return the total successor of the given nodes
                    and eids are invalid in this way.
            return_eids: If True return nodes together with corresponding eid
        """
        return self._g.successor(edge_type, nodes, return_eids)

    def sample_successor(self,
                         edge_type,
                         nodes,
                         max_degree,
                         return_eids=False,
                         shuffle=False):
        """Sample successors of given nodes with the specified edge_type.
        Args:
            edge_type: The specified edge_type.
            nodes: Given nodes whose successors will be sampled.
            max_degree: The max sampled successors for each nodes.
            return_eids: Whether to return the corresponding eids.
        Return:
            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled successor ids for given nodes with specified edge type. 
            If :code:`return_eids=True`, there will be an additional list of 
            numpy.ndarray and each numpy.ndarray represent a list of eids that 
            connected nodes to their successors.
        """
        return self._g.sample_successor(edge_type,nodes, max_degree,return_eids,shuffle)

    def predecessor(self, edge_type, nodes=None, return_eids=False):
        """Find predecessor of given nodes with the specified edge_type.
        Args:
            nodes: Return the predecessor of given nodes,
                   if nodes is None, return predecessor for all nodes
            edge_types: Return the predecessor with specified edge_type.
            return_eids: If True return nodes together with corresponding eid
        """
        return self._g.predecessor(edge_type, nodes, return_eids)

    def sample_predecessor(self,
                           edge_type,
                           nodes,
                           max_degree,
                           return_eids=False,
                           shuffle=False):
        """Sample predecessors of given nodes with the specified edge_type.
        Args:
            edge_type: The specified edge_type.
            nodes: Given nodes whose predecessors will be sampled.
            max_degree: The max sampled predecessors for each nodes.
            return_eids: Whether to return the corresponding eids.
        Return:
            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled predecessor ids for given nodes with specified edge type. 
            If :code:`return_eids=True`, there will be an additional list of 
            numpy.ndarray and each numpy.ndarray represent a list of eids that 
            connected nodes to their predecessors.
        """
        return self._g.sample_predecessor(edge_type,nodes,max_degree,return_eids,shuffle)

    def node_batch_iter(self, batch_size, shuffle=True, n_type=None):
        """Node batch iterator
        Iterate all nodes by batch with the specified node type.
        Args:
            batch_size: The batch size of each batch of nodes.
            shuffle: Whether shuffle the nodes.
            
            n_type: Iterate the nodes with the specified node type. If n_type is None, 
                    iterate all nodes by batch.
        Return:
            Batch iterator
        """
        self._g.node_batch_iter(batch_size, shuffle, n_type)

    def sample_nodes(self, sample_num, n_type=None):
        """Sample nodes with the specified n_type from the graph
        This function helps to sample nodes with the specified n_type from the graph.
        If n_type is None, this function will sample nodes from all nodes.
        Nodes might be duplicated.
        Args:
            sample_num: The number of samples
            n_type: The nodes of type to be sampled
        Return:
            A list of nodes
        """
        return self._g.sample_nodes(sample_num, n_type)

    def node_feat_info(self):
        """Return the information of node feature for HeterGraphWrapper.
        This function return the information of node features of all node types. And this
        function is used to help constructing HeterGraphWrapper
        Return:
            A list of tuple (name, shape, dtype) for all given node feature.
        """

        return self._g.node_feat_info()

    def edge_feat_info(self):
        """Return the information of edge feature for HeterGraphWrapper.
        This function return the information of edge features of all edge types. And this
        function is used to help constructing HeterGraphWrapper
        Return:
            A dict of list of tuple (name, shape, dtype) for all given edge feature.
        """
        return self._g.edge_feat_info()

    def edge_types_info(self):
        """Return the information of all edge types.
        
        Return:
            A list of all edge types.
        
        """

        return self._g.edge_types_info()