# Copyright (c) 2021 OpenKS Authors, Dlib Lab, Peking University. 
# All Rights Reserved.

"""
Loader for generating graph data
"""
import os
import json
import numpy as np

class GraphLoaderForGCN:
    """
    Specific loader for generating graph
    """
    def __init__(self, data_dir="openks/data/company-kg", directed=True):
        self.data_dir = data_dir
        self.directed = directed
        self._load_graph()
        
    def _load_graph(self):
        node_lines = open(self.data_dir + "/entities").readlines()
        self.node_name = {}
        self.node_type = {}
        for l in node_lines:
            words = l[:-1].split("\t")
            node, typ, name = words[0], words[1], words[2]
            self.node_type[int(node)] = typ
            self.node_name[int(node)] = name
        self.adj_list = {}
        self.node_num = len(self.node_name)
        #self.adj_mat = np.zeros((self.node_num, self.node_num))
        self.edge_type = {}
        edge_lines = open(self.data_dir + "/triples").readlines()
        for l in edge_lines:
            words = l.split("\t")
            head = int(words[0])
            relation = words[1]
            tail = int(words[2])
            if self.directed:
                if head not in self.adj_list:
                    self.adj_list[head] = []
                self.adj_list[head].append(tail)
                #self.adj_mat[head][tail] = 1
                self.edge_type[(head, tail)] = relation
            else:
                if head not in self.adj_list:
                    self.adj_list[head] = []
                if tail not in self.adj_list:
                    self.adj_list[tail] = []
                self.adj_list[head].append(tail)
                self.adj_list[tail].append(head)
                #self.adj_mat[head][tail] = 1
                #self.adj_mat[tail][head] = 1
                self.edge_type[(head, tail)] = relation
                self.edge_type[(tail, head)] = relation
                
    def get_node_num(self):
        return self.node_num
    
    def get_node_names(self):
        return self.node_name
    
    def get_node_types(self):
        return self.node_type
    
    def get_adj_list(self):
        return self.adj_list
    
    def get_edge_types(self):
        return self.edge_type
