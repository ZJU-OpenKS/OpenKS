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


class Graph(graph_base):
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
        if node_feat is not None:
            self._node_feat = node_feat
        else:
            self._node_feat = {}

        if edge_feat is not None:
            self._edge_feat = edge_feat
        else:
            self._edge_feat = {}

        if isinstance(edges, np.ndarray):
            if edges.dtype != "int64":
                edges = edges.astype("int64")
        else:
            edges = np.array(edges, dtype="int64")

        self._edges = edges
        self._num_nodes = num_nodes

        self._adj_src_index = None
        self._adj_dst_index = None
        self.indegree()
        self._num_graph = 1
        self._graph_lod = np.array([0, self.num_nodes], dtype="int32")

    def dump(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'num_nodes.npy'), self._num_nodes)
        np.save(os.path.join(path, 'edges.npy'), self._edges)

        if self._adj_src_index:
            self._adj_src_index.dump(os.path.join(path, 'adj_src'))

        if self._adj_dst_index:
            self._adj_dst_index.dump(os.path.join(path, 'adj_dst'))

        def dump_feat(feat_path, feat):
            """Dump all features to .npy file.
            """
            if len(feat) == 0:
                return
            if not os.path.exists(feat_path):
                os.makedirs(feat_path)
            for key in feat:
                np.save(os.path.join(feat_path, key + ".npy"), feat[key])

        dump_feat(os.path.join(path, "node_feat"), self.node_feat)
        dump_feat(os.path.join(path, "edge_feat"), self.edge_feat)

    @property
    def adj_src_index(self):
        """Return an EdgeIndex object for src.
        """
        if self._adj_src_index is None:
            if len(self._edges) == 0:
                u = np.array([], dtype="int64")
                v = np.array([], dtype="int64")
            else:
                u = self._edges[:, 0]
                v = self._edges[:, 1]

            self._adj_src_index = EdgeIndex(
                u=u, v=v, num_nodes=self._num_nodes)
        return self._adj_src_index

    @property
    def adj_dst_index(self):
        """Return an EdgeIndex object for dst.
        """
        if self._adj_dst_index is None:
            if len(self._edges) == 0:
                v = np.array([], dtype="int64")
                u = np.array([], dtype="int64")
            else:
                v = self._edges[:, 0]
                u = self._edges[:, 1]

            self._adj_dst_index = EdgeIndex(
                u=u, v=v, num_nodes=self._num_nodes)
        return self._adj_dst_index

    @property
    def edge_feat(self):
        """Return a dictionary of edge features.
        """
        return self._edge_feat

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        return self._node_feat

    @property
    def num_edges(self):
        """Return the number of edges.
        """
        return len(self._edges)

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._num_nodes

    @property
    def edges(self):
        """Return all edges in numpy.ndarray with shape (num_edges, 2).
        """
        return self._edges

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
        if sort_by not in ["src", "dst"]:
            raise ValueError("sort_by should be in 'src' or 'dst'.")
        if sort_by == 'src':
            src, dst, eid = self.adj_src_index.triples()
        else:
            dst, src, eid = self.adj_dst_index.triples()
        return src, dst, eid

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        return np.arange(self._num_nodes, dtype="int64")

    def indegree(self, nodes=None):
        """Return the indegree of the given nodes

        This function will return indegree of given nodes.

        Args:
            nodes: Return the indegree of given nodes,
                   if nodes is None, return indegree for all nodes

        Return:
            A numpy.ndarray as the given nodes' indegree.
        """
        if nodes is None:
            return self.adj_dst_index.degree
        else:
            return self.adj_dst_index.degree[nodes]

    def outdegree(self, nodes=None):
        """Return the outdegree of the given nodes.

        This function will return outdegree of given nodes.

        Args:
            nodes: Return the outdegree of given nodes,
                   if nodes is None, return outdegree for all nodes

        Return:
            A numpy.array as the given nodes' outdegree.
        """
        if nodes is None:
            return self.adj_src_index.degree
        else:
            return self.adj_src_index.degree[nodes]

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
        if return_eids:
            return self.adj_src_index.view_v(
                nodes), self.adj_src_index.view_eid(nodes)
        else:
            return self.adj_src_index.view_v(nodes)

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

        node_succ = self.successor(nodes, return_eids=return_eids)
        if return_eids:
            node_succ, node_succ_eid = node_succ

        if nodes is None:
            nodes = self.nodes

        node_succ = node_succ.tolist()

        if return_eids:
            node_succ_eid = node_succ_eid.tolist()

        if return_eids:
            return graph_kernel.sample_subset_with_eid(
                node_succ, node_succ_eid, max_degree, shuffle)
        else:
            return graph_kernel.sample_subset(node_succ, max_degree, shuffle)

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
        if return_eids:
            return self.adj_dst_index.view_v(
                nodes), self.adj_dst_index.view_eid(nodes)
        else:
            return self.adj_dst_index.view_v(nodes)

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
        node_pred = self.predecessor(nodes, return_eids=return_eids)
        if return_eids:
            node_pred, node_pred_eid = node_pred

        if nodes is None:
            nodes = self.nodes

        node_pred = node_pred.tolist()

        if return_eids:
            node_pred_eid = node_pred_eid.tolist()

        if return_eids:
            return graph_kernel.sample_subset_with_eid(
                node_pred, node_pred_eid, max_degree, shuffle)
        else:
            return graph_kernel.sample_subset(node_pred, max_degree, shuffle)

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
        node_feat_info = []
        for key, value in self._node_feat.items():
            node_feat_info.append(
                (key, _hide_num_nodes(value.shape), value.dtype))
        return node_feat_info

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
        edge_feat_info = []
        for key, value in self._edge_feat.items():
            edge_feat_info.append(
                (key, _hide_num_nodes(value.shape), value.dtype))
        return edge_feat_info

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
        reindex = {}

        for ind, node in enumerate(nodes):
            reindex[node] = ind

        if eid is None and edges is None:
            raise ValueError("Eid and edges can't be None at the same time.")

        sub_edge_feat = {}
        if edges is None:
            edges = self._edges[eid]
        else:
            edges = np.array(edges, dtype="int64")

        if with_edge_feat:
            for key, value in self._edge_feat.items():
                if eid is None:
                    raise ValueError(
                        "Eid can not be None with edge features.")
                sub_edge_feat[key] = value[eid]

        if edge_feats is not None:
            sub_edge_feat.update(edge_feats)
            
        sub_edges = graph_kernel.map_edges(
            np.arange(
                len(edges), dtype="int64"), edges, reindex)

        sub_node_feat = {}
        if with_node_feat:
            for key, value in self._node_feat.items():
                sub_node_feat[key] = value[nodes]

        subgraph = SubGraph(
            num_nodes=len(nodes),
            edges=sub_edges,
            node_feat=sub_node_feat,
            edge_feat=sub_edge_feat,
            reindex=reindex)
        return subgraph

    def node_batch_iter(self, batch_size, shuffle=True):
        """Node batch iterator

        Iterate all node by batch.

        Args:
            batch_size: The batch size of each batch of nodes.

            shuffle: Whether shuffle the nodes.

        Return:
            Batch iterator
        """
        perm = np.arange(self._num_nodes, dtype="int64")
        if shuffle:
            np.random.shuffle(perm)
        start = 0
        while start < self._num_nodes:
            yield perm[start:start + batch_size]
            start += batch_size

    def sample_nodes(self, sample_num):
        """Sample nodes from the graph

        This function helps to sample nodes from all nodes.
        Nodes might be duplicated.

        Args:
            sample_num: The number of samples

        Return:
            A list of nodes
        """
        return np.random.randint(low=0, high=self._num_nodes, size=sample_num)

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

        sampled_eid = np.random.choice(
            np.arange(self._edges.shape[0]), sample_num, replace=replace)
        return self._edges[sampled_eid], sampled_eid

    def has_edges_between(self, u, v):
        """Check whether some edges is in graph.

        Args:
            u: a numpy.array of src nodes ID.
            v: a numpy.array of dst nodes ID.

        Return:
            exists: A numpy.array of bool, with the same shape with `u` and `v`,
                exists[i] is True if (u[i], v[i]) is a edge in graph, Flase otherwise.
        """
        assert u.shape[0] == v.shape[0], "u and v must have the same shape"
        exists = np.logical_and(u < self.num_nodes, v < self.num_nodes)
        exists_idx = np.arange(u.shape[0])[exists]
        for idx, succ in zip(exists_idx, self.successor(u[exists])):
            exists[idx] = v[idx] in succ
        return exists

    def random_walk(self, nodes, max_depth):
        """Implement of random walk.

        This function get random walks path for given nodes and depth.

        Args:
            nodes: Walk starting from nodes
            max_depth: Max walking depth

        Return:
            A list of walks.
        """
        walk = []
        # init
        for node in nodes:
            walk.append([node])

        cur_walk_ids = np.arange(0, len(nodes))
        cur_nodes = np.array(nodes)
        for l in range(max_depth):
            # select the walks not end
            outdegree = self.outdegree(cur_nodes)
            mask = (outdegree != 0)
            if np.any(mask):
                cur_walk_ids = cur_walk_ids[mask]
                cur_nodes = cur_nodes[mask]
                outdegree = outdegree[mask]
            else:
                # stop when all nodes have no successor
                break
            succ = self.successor(cur_nodes)
            sample_index = np.floor(
                np.random.rand(outdegree.shape[0]) * outdegree).astype("int64")

            nxt_cur_nodes = []
            for s, ind, walk_id in zip(succ, sample_index, cur_walk_ids):
                walk[walk_id].append(s[ind])
                nxt_cur_nodes.append(s[ind])
            cur_nodes = np.array(nxt_cur_nodes)
        return walk

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
        if p == 1. and q == 1.:
            return self.random_walk(nodes, max_depth)

        walk = []
        # init
        for node in nodes:
            walk.append([node])

        cur_walk_ids = np.arange(0, len(nodes))
        cur_nodes = np.array(nodes)
        prev_nodes = np.array([-1] * len(nodes), dtype="int64")
        prev_succs = np.array([[]] * len(nodes), dtype="int64")
        for l in range(max_depth):
            # select the walks not end
            outdegree = self.outdegree(cur_nodes)
            mask = (outdegree != 0)
            if np.any(mask):
                cur_walk_ids = cur_walk_ids[mask]
                cur_nodes = cur_nodes[mask]
                prev_nodes = prev_nodes[mask]
                prev_succs = prev_succs[mask]
            else:
                # stop when all nodes have no successor
                break
            cur_succs = self.successor(cur_nodes)
            num_nodes = cur_nodes.shape[0]
            nxt_nodes = np.zeros(num_nodes, dtype="int64")

            for idx, (succ, prev_succ, walk_id, prev_node) in enumerate(
                    zip(cur_succs, prev_succs, cur_walk_ids, prev_nodes)):

                sampled_succ = graph_kernel.node2vec_sample(succ, prev_succ,
                                                            prev_node, p, q)
                walk[walk_id].append(sampled_succ)
                nxt_nodes[idx] = sampled_succ

            prev_nodes, prev_succs = cur_nodes, cur_succs
            cur_nodes = nxt_nodes
        return walk

    @property
    def num_graph(self):
        """ Return Number of Graphs"""
        return self._num_graph

    @property
    def graph_lod(self):
        """ Return Graph Lod Index for Paddle Computation"""
        return self._graph_lod




class HeterGraph(graph_base):
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
        self._num_nodes = num_nodes
        self._edges_dict = edges

        if isinstance(node_types, list):
            self._node_types = np.array(node_types, dtype=object)[:, 1]
        else:
            self._node_types = node_types

        self._nodes_type_dict = {}
        for n_type in np.unique(self._node_types):
            self._nodes_type_dict[n_type] = np.where(
                self._node_types == n_type)[0]

        if node_feat is not None:
            self._node_feat = node_feat
        else:
            self._node_feat = {}

        if edge_feat is not None:
            self._edge_feat = edge_feat
        else:
            self._edge_feat = {}

        self._multi_graph = {}

        for key, value in self._edges_dict.items():
            if not self._edge_feat:
                edge_feat = None
            else:
                edge_feat = self._edge_feat[key]

            self._multi_graph[key] = Graph(
                num_nodes=self._num_nodes,
                edges=value,
                node_feat=self._node_feat,
                edge_feat=edge_feat)

        self._edge_types = self.edge_types_info()

    def dump(self, path, indegree=False, outdegree=False):

        if indegree:
            for e_type, g in self._multi_graph.items():
                g.indegree()

        if outdegree:
            for e_type, g in self._multi_graph.items():
                g.outdegree()

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, "num_nodes.npy"), self._num_nodes)
        np.save(os.path.join(path, "node_types.npy"), self._node_types)
        with open(os.path.join(path, "edge_types.pkl"), 'wb') as f:
            pkl.dump(self._edge_types, f)
        with open(os.path.join(path, "nodes_type_dict.pkl"), 'wb') as f:
            pkl.dump(self._nodes_type_dict, f)

        for e_type, g in self._multi_graph.items():
            sub_path = os.path.join(path, e_type)
            g.dump(sub_path)

    @property
    def edge_types(self):
        """Return a list of edge types.
        """
        return self._edge_types

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._num_nodes

    @property
    def num_edges(self):
        """Return edges number of all edge types.
        """
        n_edges = {}
        for e_type in self._edge_types:
            n_edges[e_type] = self._multi_graph[e_type].num_edges
        return n_edges

    @property
    def node_types(self):
        """Return the node types.
        """
        return self._node_types

    @property
    def edge_feat(self, edge_type=None):
        """Return edge features of all edge types.
        """
        return self._edge_feat

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        return self._node_feat

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        return np.arange(self._num_nodes, dtype='int64')

    def __getitem__(self, edge_type):
        """__getitem__
        """
        return self._multi_graph[edge_type]

    def num_nodes_by_type(self, n_type=None):
        """Return the number of nodes with the specified node type.
        """
        if n_type not in self._nodes_type_dict:
            raise ("%s is not in valid node type" % n_type)
        else:
            return len(self._nodes_type_dict[n_type])

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
        if edge_type is None:
            indegrees = []
            for e_type in self._edge_types:
                indegrees.append(self._multi_graph[e_type].indegree(nodes))
            indegrees = np.sum(np.vstack(indegrees), axis=0)
            return indegrees
        else:
            return self._multi_graph[edge_type].indegree(nodes)

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
        if edge_type is None:
            outdegrees = []
            for e_type in self._edge_types:
                outdegrees.append(self._multi_graph[e_type].outdegree(nodes))
            outdegrees = np.sum(np.vstack(outdegrees), axis=0)
            return outdegrees
        else:
            return self._multi_graph[edge_type].outdegree(nodes)

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
        return self._multi_graph[edge_type].successor(nodes, return_eids)

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
        return self._multi_graph[edge_type].sample_successor(
            nodes=nodes,
            max_degree=max_degree,
            return_eids=return_eids,
            shuffle=shuffle)

    def predecessor(self, edge_type, nodes=None, return_eids=False):
        """Find predecessor of given nodes with the specified edge_type.

        Args:
            nodes: Return the predecessor of given nodes,
                   if nodes is None, return predecessor for all nodes

            edge_types: Return the predecessor with specified edge_type.

            return_eids: If True return nodes together with corresponding eid
        """
        return self._multi_graph[edge_type].predecessor(nodes, return_eids)

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
        return self._multi_graph[edge_type].sample_predecessor(
            nodes=nodes,
            max_degree=max_degree,
            return_eids=return_eids,
            shuffle=shuffle)

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
        if n_type is None:
            nodes = np.arange(self._num_nodes, dtype="int64")
        else:
            nodes = self._nodes_type_dict[n_type]

        if shuffle:
            np.random.shuffle(nodes)
        start = 0
        while start < len(nodes):
            yield nodes[start:start + batch_size]
            start += batch_size

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
        if n_type is not None:
            return np.random.choice(
                self._nodes_type_dict[n_type], size=sample_num)
        else:
            return np.random.randint(
                low=0, high=self._num_nodes, size=sample_num)

    def node_feat_info(self):
        """Return the information of node feature for HeterGraphWrapper.

        This function return the information of node features of all node types. And this
        function is used to help constructing HeterGraphWrapper

        Return:
            A list of tuple (name, shape, dtype) for all given node feature.

        """
        node_feat_info = []
        for feat_name, feat in self._node_feat.items():
            node_feat_info.append(
                (feat_name, _hide_num_nodes(feat.shape), feat.dtype))

        return node_feat_info

    def edge_feat_info(self):
        """Return the information of edge feature for HeterGraphWrapper.

        This function return the information of edge features of all edge types. And this
        function is used to help constructing HeterGraphWrapper

        Return:
            A dict of list of tuple (name, shape, dtype) for all given edge feature.

        """
        edge_feat_info = {}
        for edge_type_name, feat_dict in self._edge_feat.items():
            tmp_edge_feat_info = []
            for feat_name, feat in feat_dict.items():
                full_name = feat_name
                tmp_edge_feat_info.append(
                    (full_name, _hide_num_nodes(feat.shape), feat.dtype))
            edge_feat_info[edge_type_name] = tmp_edge_feat_info
        return edge_feat_info

    def edge_types_info(self):
        """Return the information of all edge types.
        
        Return:
            A list of all edge types.
        
        """
        edge_types_info = []
        for key, _ in self._multi_graph.items():
            edge_types_info.append(key)

        return edge_types_info
