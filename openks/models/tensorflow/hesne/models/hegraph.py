import copy

class HeGraph():
    def __init__(self, params, train_data, valid_data, test_data):
        self.params = params
        # self.nodes = [[] for type in range(self.params['node_type_numbers'])]
        # self.nodes_degree_train = [{} for type in range(self.params['node_type_numbers'])]
        self.type_num = params['node_type_numbers']
        self.nodes_degree_cur = [{} for _ in range(self.type_num)]
        self.nodes_degree = [{} for _ in range(self.type_num)]
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        # self.event_type = {}

    def build_hegraph(self):
        for event in self.train_data:
            # self.event_type[event['type']] = self.event_type.get(event['type'], 1) + 1
            for type in range(self.type_num):
                for node in event[type]:
                    self.nodes_degree[type][node] = self.nodes_degree[type].get(node, 0) + 1
        #49196 23672 37415


        for event in self.valid_data:
            # self.event_type[event['type']] = self.event_type.get(event['type'], 1) + 1
            for type in range(self.type_num):
                for node in event[type]:
                    self.nodes_degree[type][node] = self.nodes_degree[type].get(node, 0) + 1
        #50797 24434 37474

        for event in self.test_data:
            # self.event_type[event['type']] = self.event_type.get(event['type'], 1) + 1
            for type in range(self.type_num):
                for node in event[type]:
                    self.nodes_degree[type][node] = self.nodes_degree[type].get(node, 0) + 1
        #51589 24857 37493

    def get_totalnum_pertype(self):
        totalnum_pertype = []
        for type in range(self.type_num):
            totalnum_pertype.append(len(self.nodes_degree[type].keys()))
        return totalnum_pertype

    # def get_curdegree_pertype(self, batch_data):
    #     nodes_degree_last = copy.deepcopy(self.nodes_degree_cur)
    #     for event in range(self.params['batch_event_numbers']):
    #         for type in range(self.params['node_type_numbers']):
    #             for node in batch_data[event][type]:
    #                 self.nodes_degree_cur[type][node] = self.nodes_degree_cur[type].get(node, 1) + 1
    #                 # if node not in self.nodes_degree_cur[type]:
    #                 #     self.nodes_degree_cur[type][node] = 1
    #                 # else:
    #                 #     self.nodes_degree_cur[type][node] += 1
    #     return nodes_degree_last, self.nodes_degree_cur
        # return self.nodes_degree_cur

    # def get_eventtype_num(self):
    #     return len(self.event_type.keys())

    # def savegraph(self, graph_dir):
    #     num_pertype = self.get_totalnum_pertype()
    #     total_num = sum(num_pertype)
    #     event_num = 0
    #     with open(graph_dir, 'w') as graph_data:
    #         for event in self.train_data:
    #             for type in range(self.type_num):
    #                 if type == 0:
    #                     for node in event[type]:
    #                         graph_data.write(str(event_num+total_num)+'\t'+str(node)+'\n')
    #                 else:
    #                     for node in event[type]:
    #                         for pretype in range(type):
    #                             node += num_pertype[pretype]
    #                         graph_data.write(str(event_num+total_num)+'\t'+str(node)+'\n')
    #             event_num += 1












