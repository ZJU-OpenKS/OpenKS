import torch
import time
import numpy as np
import dgl
from collections import defaultdict
from torch.utils.data import Dataset

class S2orcDataset(Dataset):
    def __init__(self, heter_graph, section_withtexts, citation_edges, citation_pos_all, vocab2index, args):
        super(S2orcDataset, self).__init__()
        self.heter_graph = heter_graph
        self.section_withtexts = section_withtexts
        self.citation_edges = citation_edges
        self.citation_pos_all = citation_pos_all
        self.vocab2index = vocab2index
        self.is_pointer_gen = args.is_pointer_gen
        self.n_max_len_section = args.n_max_len_section
        self.n_max_len_citation = args.n_max_len_citation
        self.n_neigh_layer = args.n_neigh_layer
        self.n_max_neigh = args.n_max_neigh
        self.n_neg = args.n_neg

    def get_subgraph_from_heterograph(self, seed_node_1, seed_node_2, k_hops, fanout):
        # distances = defaultdict(dict)
        distances = {}
        for node_type in self.heter_graph.ntypes:
            distances[node_type] = torch.ones(self.heter_graph.num_nodes(node_type), 2) * (k_hops+1)
        idx = 0
        nodes_subgraph_1 = defaultdict(set)
        nodes_subgraph_2 = defaultdict(set)

        for nodes_subgraph, seed_node in zip([nodes_subgraph_1, nodes_subgraph_2], [seed_node_1, seed_node_2]):
            subgraph_in = dgl.sampling.sample_neighbors(self.heter_graph, nodes=seed_node, fanout=fanout[0], edge_dir='out')
            for node_type, node_ids in seed_node.items():
                distances[node_type][node_ids, idx] = 0
                nodes_subgraph[node_type].update(node_ids)
            for k_hop in range(k_hops):
                subgraph = subgraph_in
                new_adj_nodes = defaultdict(set)
                for node_type_1, edge_type, node_type_2 in subgraph.canonical_etypes:
                    if idx == 0 and k_hop == 0 and ('cited' in edge_type):
                        continue #avoid src has citing papers
                    nodes_id_1, nodes_id_2 = subgraph.all_edges(etype=edge_type)
                    new_adj_nodes[node_type_1].update(set(nodes_id_1.numpy()).difference(nodes_subgraph[node_type_1]))
                    new_adj_nodes[node_type_2].update(set(nodes_id_2.numpy()).difference(nodes_subgraph[node_type_2]))
                    nodes_subgraph[node_type_1].update(new_adj_nodes[node_type_1])
                    nodes_subgraph[node_type_2].update(new_adj_nodes[node_type_2])

                new_adj_nodes = {key: list(value) for key, value in new_adj_nodes.items()}

                subgraph_in = dgl.sampling.sample_neighbors(self.heter_graph, nodes=dict(new_adj_nodes), fanout=fanout[k_hop],
                                                            edge_dir='out')
                for node_type, node_ids in new_adj_nodes.items():
                    distances[node_type][node_ids, idx] = k_hop + 1

            idx += 1
        #merge nodes_subgraph_1 and nodes_subgraph_2
        nodes_subgraph = {}
        for node_type in nodes_subgraph_2.keys():
            nodes_subgraph[node_type] = list(nodes_subgraph_1[node_type].union(nodes_subgraph_2[node_type]))
        return nodes_subgraph, distances

    # def get_bow_representation(self, text_sequence):
    #     sequence_bow_rep = np.zeros(shape=len(self.vocab2index), dtype=np.float32)
    #     for word in text_sequence:
    #         if word in self.vocab2index:
    #             index = self.vocab2index[word]
    #             sequence_bow_rep[index] += 1
    #     sequence_bow_rep /= np.max([np.sum(sequence_bow_rep), 1])
    #     return torch.tensor(np.asarray(sequence_bow_rep))

    def __getitem__(self, index):
        # index = 1435
        current_pos = self.citation_edges[index]['cite_label'] - 1
        src = self.citation_edges[index]['src']
        dst = self.citation_edges[index]['dst']
        citation_pos_all = list(self.citation_pos_all[f'{src}_{dst}'])
        citation_pos = np.zeros(shape=4, dtype=np.float32)
        for pos in citation_pos_all:
            citation_pos[pos-1] = 1
        citation_pos = torch.Tensor(citation_pos)

        seed_node_1 = {'paper': [src]}
        seed_node_2 = {'paper': [dst]}
        nodes_subgraph, distances = \
            self.get_subgraph_from_heterograph(seed_node_1, seed_node_2, self.n_neigh_layer, self.n_max_neigh)
        for node_type in self.heter_graph.ntypes:
            self.heter_graph.nodes[node_type].data['center_dist'] = distances[node_type]
        sub_g = self.heter_graph.subgraph(nodes_subgraph)

        def nodes_with_source(nodes):
            return (nodes.data['node_id'] == src)
        def nodes_with_target(nodes):
            return (nodes.data['node_id'] == dst)

        source_id = sub_g.filter_nodes(nodes_with_source, ntype='paper')
        target_id = sub_g.filter_nodes(nodes_with_target, ntype='paper')
        pair_id = torch.cat([source_id, target_id], dim=0)

        for pos in citation_pos_all:
            if pos == 1:
                citation_type = 'intro'
            elif pos == 2:
                citation_type = 'relate'
            elif pos == 3:
                citation_type = 'method'
            else:
                citation_type = 'experiment'
            if sub_g.has_edges_between(source_id, target_id, etype='citing-'+citation_type):
                citing_edge_id = sub_g.edge_ids(source_id, target_id, etype='citing-'+citation_type)
                cited_edge_id = sub_g.edge_ids(target_id, source_id, etype='cited-'+citation_type)
                sub_g.remove_edges(eids=citing_edge_id, etype='citing-'+citation_type)
                sub_g.remove_edges(eids=cited_edge_id, etype='cited-'+citation_type)

        sub_g_oovs = {}
        enc_extend_vocab = {}
        for sec_type in ['intro', 'method', 'experiment']:
            sec_texts_extend_vocab = []
            for pidx in [src, dst]:
                sec_text_ = self.section_withtexts[pidx][sec_type]
                if len(sec_text_) > self.n_max_len_section:
                    sec_text_ = sec_text_[:self.n_max_len_section]
                sec_text_extend_vocab = []
                for word in sec_text_:
                    if word in self.vocab2index:
                        sec_text_extend_vocab.append(self.vocab2index[word])
                    else:
                        if word not in sub_g_oovs:
                            sub_g_oovs[word] = len(self.vocab2index) + len(sub_g_oovs)
                        sec_text_extend_vocab.append(sub_g_oovs[word])
                sec_text_extend_vocab.extend([0] * (self.n_max_len_section - len(sec_text_extend_vocab)))
                sec_texts_extend_vocab.append(torch.LongTensor(sec_text_extend_vocab))
            enc_extend_vocab[sec_type] = torch.stack(sec_texts_extend_vocab, dim=0)

            sec_texts = []
            for pidx in sub_g.nodes['paper'].data['node_id'].tolist():
                sec_text_ = self.section_withtexts[pidx][sec_type]
                if len(sec_text_) > self.n_max_len_section:
                    sec_text_ = sec_text_[:self.n_max_len_section] # TODO: truncate seq from two sides for section text
                sec_text = []
                for word in sec_text_:
                    if word in self.vocab2index:
                        sec_text.append(self.vocab2index[word])
                    else:
                        sec_text.append(self.vocab2index['UNK'])
                sec_text.extend([0] * (self.n_max_len_section - len(sec_text)))
                sec_texts.append(torch.LongTensor(sec_text))

            sec_texts = torch.stack(sec_texts, dim=0)
            sub_g.nodes['paper'].data[sec_type+'_text'] = sec_texts

        # citation text
        citation_text_ = self.citation_edges[index]['cite_text']
        citation_text = []
        citation_text_extend_vocab = []
        for word in citation_text_:
            if word in self.vocab2index:
                citation_text.append(self.vocab2index[word])
                citation_text_extend_vocab.append(self.vocab2index[word])
            else:
                if word in sub_g_oovs:
                    citation_text_extend_vocab.append(sub_g_oovs[word])
                else:
                    citation_text_extend_vocab.append(self.vocab2index['UNK'])
                citation_text.append(self.vocab2index['UNK'])
        citation_text_inp = [self.vocab2index['SOS']] + citation_text[:-1]
        if self.is_pointer_gen:
            citation_text_target = citation_text_extend_vocab
        else:
            citation_text_target = citation_text

        # citation_bow = self.get_bow_representation(citation_text_[:self.n_max_len_citation])

        # if len(citation_text_inp) >= self.n_max_len_citation:
        #     citation_text_inp = citation_text_inp[:self.n_max_len_citation]
        #     citation_text_target = citation_text_target[:self.n_max_len_citation] #TODO: truncate seq from two sides
        # else:
        #     citation_text_inp = [self.vocab2index['SOS']] + citation_text
        #     citation_text_target.append(self.vocab2index['EOS'])

        if len(citation_text_inp) > self.n_max_len_citation:
            citation_text_inp = citation_text_inp[:self.n_max_len_citation]
            citation_text_target = citation_text_target[:self.n_max_len_citation] #TODO: truncate seq from two sides
        else:
            citation_text_target.append(self.vocab2index['EOS'])

        citation_text_inp = torch.LongTensor(citation_text_inp)
        citation_text_target = torch.LongTensor(citation_text_target)

        # get shuf paper neg
        shuf_paper = torch.stack([torch.randperm(sub_g.num_nodes('paper'))
                                  for _ in range(self.n_neg)], dim=1)
        # 2 * (3 * seq_len)
        enc_extend_vocab = torch.cat(list(enc_extend_vocab.values()), dim=1)
        enc_extend_vocab = torch.cat([enc_extend_vocab[0, :], enc_extend_vocab[1, :]], dim=0)
        # enc_extend_vocab = torch.stack(list(enc_extend_vocab.values()), dim=1)

        # index2vocab = {idx: w for w, idx in self.vocab2index.items()}
        # print(sub_g.nodes['paper'].data['intro_text'].shape)
        # texts = sub_g.nodes['paper'].data['intro_text'].tolist()
        # for text in texts:
        #     print([index2vocab[i] for i in text])
        # print('--------------------------------')
        # print(sub_g.nodes['paper'].data['method_text'])
        # print('--------------------------------')
        # print(sub_g.nodes['paper'].data['experiment_text'])
        # print('--------------------------------')
        # print(enc_extend_vocab)
        # print('--------------------------------')
        # print([index2vocab[i] for i in citation_text_target])
        # time.sleep(100)
        pair_ogid = [src, dst]

        # print(current_pos)
        # print(citation_pos_all)
        # time.sleep(2)

        return sub_g, pair_id, current_pos, citation_pos, shuf_paper, enc_extend_vocab, \
               citation_text_inp, citation_text_target, sub_g_oovs, pair_ogid

    def __len__(self):
        return len(self.citation_edges)