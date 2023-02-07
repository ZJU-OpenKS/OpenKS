import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mutual_info import MutualInfoMax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

class GRUEncoder(nn.Module):
    def __init__(self, n_dim, n_layers, drop_emb, dropout=0.35):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(n_dim, n_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.drop_emb = drop_emb
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emb, seq_len, idx_unsort):
        if self.drop_emb:
            seq_emb = self.dropout(seq_emb)
        packed = pack_padded_sequence(seq_emb, seq_len, batch_first=True, enforce_sorted=False)
        output, h_last = self.gru(packed)
        output = pad_packed_sequence(output, batch_first=True)

        # output = output[0].index_select(0, idx_unsort)
        # h_last = h_last.transpose(0, 1)
        # h_last = h_last.index_select(0, idx_unsort)

        output = output[0]
        h_last = h_last.transpose(0, 1)

        return output, h_last

class HGTLayer_inter(nn.Module):
    def __init__(self, device, n_hid, section_type2index, dropout=0.2, use_norm=False):
        super(HGTLayer_inter, self).__init__()
        self.n_hid = n_hid
        self.section_type2index = section_type2index
        self.section_index2type = {v: k for k, v in section_type2index.items()}
        self.use_norm = use_norm

        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        if use_norm:
            self.norms = nn.ModuleDict()

        # k, q, v, norm for paper type (general intro method experiment)
        for section in ['general', 'intro', 'method', 'experiment']:
            self.k_linears.update({section: nn.Linear(n_hid//2, n_hid//2)})
            self.q_linears.update({section: nn.Linear(n_hid//2, n_hid//2)})
            self.v_linears.update({section: nn.Linear(n_hid//2, n_hid//2)})
            self.a_linears.update({section: nn.Linear(n_hid//2, n_hid//2)})
            if use_norm:
                self.norms.update({section: nn.LayerNorm(n_hid//2)})

        # k, q, v, norm for relate type
        self.k_linears.update({'relate': nn.Linear(n_hid*3//2, n_hid*3//2)})
        self.q_linears.update({'relate': nn.Linear(n_hid*3//2, n_hid*3//2)})
        self.v_linears.update({'relate': nn.Linear(n_hid*3//2, n_hid*3//2)})
        self.a_linears.update({'relate': nn.Linear(n_hid*3//2, n_hid*3//2)})
        if use_norm:
            self.norms.update({'relate': nn.LayerNorm(n_hid*3//2)})

        # skip for general intro method experiment
        self.skip = nn.Parameter(torch.ones(5))

        self.sqrt_dk = [math.sqrt(n_hid), math.sqrt(n_hid * 2)]
        self.drop = nn.Dropout(dropout)

    def edge_attention_specific(self, edges):
        sec_idx = edges.data['stype_id'][0]
        sec_type = self.section_index2type[sec_idx.item()]

        key_general = edges.src['k_general']
        val_general = edges.src['v_general']
        qry_general = edges.dst['q_general']

        key_shuf_general = edges.src['k_shuf_general']
        val_shuf_general = edges.src['v_shuf_general']
        qry_shuf_general = edges.dst['q_shuf_general']

        key_specific = edges.src['k_'+sec_type]
        val_specific = edges.src['v_'+sec_type]
        qry_specific = edges.dst['q_'+sec_type]

        key_shuf_specific = edges.src['k_shuf_'+sec_type]
        val_shuf_specific = edges.src['v_shuf_'+sec_type]
        qry_shuf_specific = edges.dst['q_shuf_'+sec_type]

        key_sec = torch.cat([key_general, key_specific], dim=-1)
        val_sec = torch.cat([val_general, val_specific], dim=-1)
        qry_sec = torch.cat([qry_general, qry_specific], dim=-1)

        # n_neg = key_shuf_general.size(1)

        key_shuf_sec = torch.cat([key_shuf_general, key_shuf_specific], dim=-1)
        val_shuf_sec = torch.cat([val_shuf_general, val_shuf_specific], dim=-1)
        qry_shuf_sec = torch.cat([qry_shuf_general, qry_shuf_specific], dim=-1)

        if sec_type == 'relate':
            sqrt_dk = self.sqrt_dk[1]
        else:
            sqrt_dk = self.sqrt_dk[0]
        att_sec = (qry_sec*key_sec).sum(dim=-1) / sqrt_dk
        att_shuf_sec = (qry_shuf_sec*key_shuf_sec).sum(dim=-1) / sqrt_dk
        return {'a_sec': att_sec, 'v_sec': val_sec,
                'a_shuf_sec': att_shuf_sec, 'v_shuf_sec': val_shuf_sec}

    def message_func_specific(self, edges):
        return {'a_sec': edges.data['a_sec'], 'v_sec': edges.data['v_sec'],
                'a_shuf_sec': edges.data['a_shuf_sec'], 'v_shuf_sec': edges.data['v_shuf_sec']}

    def reduce_func_intro(self, nodes):
        att_sec = F.softmax(nodes.mailbox['a_sec'], dim=1)
        h_sec = torch.sum(att_sec.unsqueeze(dim=-1) * nodes.mailbox['v_sec'], dim=1).chunk(2, dim=-1)
        att_shuf_sec = F.softmax(nodes.mailbox['a_shuf_sec'], dim=1)
        h_shuf_sec = torch.sum(att_shuf_sec.unsqueeze(dim=-1) * nodes.mailbox['v_shuf_sec'], dim=1).chunk(2, dim=-1)
        return {'t_general': h_sec[0], 't_shuf_general': h_shuf_sec[0], 't_intro': h_sec[1], 't_shuf_intro': h_shuf_sec[1]}

    def reduce_func_method(self, nodes):
        att_sec = F.softmax(nodes.mailbox['a_sec'], dim=1)
        h_sec = torch.sum(att_sec.unsqueeze(dim=-1) * nodes.mailbox['v_sec'], dim=1).chunk(2, dim=-1)
        att_shuf_sec = F.softmax(nodes.mailbox['a_shuf_sec'], dim=1)
        h_shuf_sec = torch.sum(att_shuf_sec.unsqueeze(dim=-1) * nodes.mailbox['v_shuf_sec'], dim=1).chunk(2, dim=-1)
        return {'t_general': h_sec[0], 't_shuf_general': h_shuf_sec[0], 't_method': h_sec[1], 't_shuf_method': h_shuf_sec[1]}

    def reduce_func_experiment(self, nodes):
        att_sec = F.softmax(nodes.mailbox['a_sec'], dim=1)
        h_sec = torch.sum(att_sec.unsqueeze(dim=-1) * nodes.mailbox['v_sec'], dim=1).chunk(2, dim=-1)
        att_shuf_sec = F.softmax(nodes.mailbox['a_shuf_sec'], dim=1)
        h_shuf_sec = torch.sum(att_shuf_sec.unsqueeze(dim=-1) * nodes.mailbox['v_shuf_sec'], dim=1).chunk(2, dim=-1)
        return {'t_general': h_sec[0], 't_shuf_general': h_shuf_sec[0], 't_experiment': h_sec[1], 't_shuf_experiment': h_shuf_sec[1]}

    # def reduce_func_relate(self, nodes):
    #     att_sec = F.softmax(nodes.mailbox['a_sec'], dim=1)
    #     h_sec = torch.sum(att_sec.unsqueeze(dim=-1) * nodes.mailbox['v_sec'], dim=1).chunk(4, dim=-1)
    #     att_shuf_sec = F.softmax(nodes.mailbox['a_shuf_sec'], dim=1)
    #     h_shuf_sec = torch.sum(att_shuf_sec.unsqueeze(dim=-1) * nodes.mailbox['v_shuf_sec'], dim=1).chunk(4, dim=-1)
    #     return {'t_general': h_sec[0], 't_shuf_general': h_shuf_sec[0],
    #             't_intro': h_sec[1], 't_method': h_sec[2], 't_experiment': h_sec[3]}

    def reduce_func_relate(self, nodes):
        att_sec = F.softmax(nodes.mailbox['a_sec'], dim=1)
        h_sec = torch.sum(att_sec.unsqueeze(dim=-1) * nodes.mailbox['v_sec'], dim=1).chunk(4, dim=-1)
        att_shuf_sec = F.softmax(nodes.mailbox['a_shuf_sec'], dim=1)
        h_shuf_sec = torch.sum(att_shuf_sec.unsqueeze(dim=-1) * nodes.mailbox['v_shuf_sec'], dim=1).chunk(4, dim=-1)
        return {'t_general': h_sec[0], 't_shuf_general': h_shuf_sec[0], 't_relate': torch.cat(h_sec[1:], dim=-1),
                't_shuf_relate': torch.cat(h_shuf_sec[1:], dim=-1)}

    def forward(self, batch_sub_g):
        for sec_type in ['general', 'intro', 'method', 'experiment', 'relate']:
            k_linear = self.k_linears[sec_type]
            v_linear = self.v_linears[sec_type]
            q_linear = self.q_linears[sec_type]

            batch_sub_g.nodes['paper'].data['k_'+sec_type] = \
                k_linear(batch_sub_g.nodes['paper'].data[sec_type+'_states'])
            batch_sub_g.nodes['paper'].data['v_'+sec_type] = \
                v_linear(batch_sub_g.nodes['paper'].data[sec_type+'_states'])
            batch_sub_g.nodes['paper'].data['q_'+sec_type] = \
                q_linear(batch_sub_g.nodes['paper'].data[sec_type+'_states'])
            # if sec_type == 'general':
            batch_sub_g.nodes['paper'].data['k_shuf_'+sec_type] = \
                k_linear(batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states'])
            batch_sub_g.nodes['paper'].data['v_shuf_'+sec_type] = \
                v_linear(batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states'])
            batch_sub_g.nodes['paper'].data['q_shuf_'+sec_type] = \
                q_linear(batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states'])

        update_dict = {}
        for srctype, etype, dsttype in batch_sub_g.canonical_etypes:
            sec_type = etype.split('-')[1]
            if batch_sub_g.num_edges(etype) < 1:
                continue
            batch_sub_g.apply_edges(func=self.edge_attention_specific, etype=etype)
            if sec_type == 'intro':
                update_dict[etype] = (self.message_func_specific, self.reduce_func_intro)
            elif sec_type == 'relate':
                update_dict[etype] = (self.message_func_specific, self.reduce_func_relate)
            elif sec_type == 'method':
                update_dict[etype] = (self.message_func_specific, self.reduce_func_method)
            else:
                update_dict[etype] = (self.message_func_specific, self.reduce_func_experiment)
        batch_sub_g.multi_update_all(update_dict, cross_reducer='mean')

        for sec_type in ['general', 'intro', 'method', 'experiment', 'relate']:
            # if sec_type == 'shuf_general':
            #     alpha = torch.sigmoid(self.skip[self.section_type2index['general']])
            #     a_linear = self.a_linears['general']
            #     norm = self.norms['general']
            # else:
            alpha = torch.sigmoid(self.skip[self.section_type2index[sec_type]])
            a_linear = self.a_linears[sec_type]
            norm = self.norms[sec_type]
            # try:
            trans_out = a_linear(batch_sub_g.nodes['paper'].data['t_'+sec_type])
            trans_out = trans_out * alpha + batch_sub_g.nodes['paper'].data[sec_type+'_states'] * (1 - alpha)
            shuf_trans_out = a_linear(batch_sub_g.nodes['paper'].data['t_shuf_'+sec_type])
            shuf_trans_out = shuf_trans_out * alpha + batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states'] * (1 - alpha)
            if self.use_norm:
                batch_sub_g.nodes['paper'].data[sec_type+'_states'] = self.drop(norm(trans_out))
                batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states'] = self.drop(norm(shuf_trans_out))
            else:
                batch_sub_g.nodes['paper'].data[sec_type+'_states'] = self.drop(trans_out)
                batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states'] = self.drop(shuf_trans_out)
            # except:
            #     continue


class HGTLayer_intra(nn.Module):
    def __init__(self, device, emb_vocab, n_hid, n_layers, is_drop_emb, dropout=0.35):
        super(HGTLayer_intra, self).__init__()
        self.device = device
        self.emb_vocab = emb_vocab
        self.hid_dim = n_hid
        self.text_encoder = GRUEncoder(n_hid, n_layers, is_drop_emb, dropout).to(device)
        self.drop = nn.Dropout(dropout)

    def forward(self, batch_sub_g):
        general_node_emb = {}
        specific_node_emb = {}
        seq_node_emb = {}
        for sec_type in ['intro', 'method', 'experiment']:
            # section_emb: (batch_size, src_len, hidden_size)
            section_emb = self.emb_vocab(batch_sub_g.nodes['paper'].data[sec_type+'_text'])
            # output_emb: (num_layers, batch_size, hidden_size)

            # section_len = (batch_sub_g.nodes['paper'].data[sec_type+'_text'] > 0).sum(dim=1)
            # _, idx_sort = torch.sort(section_len, dim=0, descending=True)
            # _, idx_unsort = torch.sort(idx_sort, dim=0)
            # section_emb = section_emb.index_select(0, idx_sort)
            # section_len = list(section_len[idx_sort])

            section_len = (batch_sub_g.nodes['paper'].data[sec_type+'_text'] > 0).sum(dim=1)
            idx_unsort = None
            out_hs, output_emb = self.text_encoder(section_emb, section_len, idx_unsort)



            seq_node_emb[sec_type] = out_hs
            # output_emb = output_emb.transpose(0, 1)
            output_emb = output_emb.chunk(2, dim=-1)
            general_node_emb[sec_type] = output_emb[0]
            specific_node_emb[sec_type] = output_emb[1]
        # (batch_size, num_layers, hidden_size * 3 // 2)
        specific_node_emb['relate'] = torch.cat(list(specific_node_emb.values()), dim=-1)
        # seq_node_emb['relate'] = torch.cat(list(seq_node_emb.values()), dim=-1)
        # (batch_size, num_layers, hidden_size // 2)
        general_emb = torch.sum(torch.stack(list(general_node_emb.values()), dim=0), dim=0)
        return general_emb, specific_node_emb, seq_node_emb

class DisenPaperEncoder(nn.Module):
    def __init__(self, device, heter_graph, section_type2index, section_fc, emb_vocab, args):
        super(DisenPaperEncoder, self).__init__()
        self.heter_graph = heter_graph
        self.section_type2index = section_type2index
        self.section_fc = section_fc
        self.n_hid = args.n_hid
        self.gcs_inter = nn.ModuleList().to(device)
        self.gcs_intra = HGTLayer_intra(device, emb_vocab, args.n_hid, args.n_textenc_layer,
                                        args.is_drop_emb, args.dropout).to(device)
        self.n_graphenc_layers = args.n_graphenc_layer
        for _ in range(args.n_graphenc_layer):
            self.gcs_inter.append(HGTLayer_inter(device, args.n_hid, section_type2index,
                                                 args.dropout, args.use_norm).to(device))
        self.use_mutual = args.use_mutual
        self.mutual_info = MutualInfoMax(device, section_fc, args.n_hid,
                                         args.alpha1, args.alpha2, args.alpha3).to(device) \
            if self.use_mutual else None

    def forward(self, batch_sub_g, batch_pair, batch_shuf_paper):
        paper_general_states, paper_specific_states, paper_seq_states = self.gcs_intra(batch_sub_g)
        batch_sub_g.nodes['paper'].data['general_states'] = paper_general_states
        batch_sub_g.nodes['paper'].data['shuf_general_states'] = paper_general_states[batch_shuf_paper]

        for section in paper_specific_states.keys():
            batch_sub_g.nodes['paper'].data[section+'_states'] = paper_specific_states[section]
            batch_sub_g.nodes['paper'].data['shuf_'+section+'_states'] = paper_specific_states[section][batch_shuf_paper]
            if section != 'relate':
                batch_sub_g.nodes['paper'].data[section+'_seq_states'] = paper_seq_states[section]

        for i in range(self.n_graphenc_layers):
            self.gcs_inter[i](batch_sub_g)

        batch_idx_s = batch_pair[:, 0]
        batch_idx_t = batch_pair[:, 1]
        general_s = batch_sub_g.nodes['paper'].data['general_states'][batch_idx_s]
        general_t = batch_sub_g.nodes['paper'].data['general_states'][batch_idx_t]
        general_pair = self.section_fc['general'](torch.cat([general_s, general_t], dim=-1))
        # get shuf neg general
        general_states = batch_sub_g.nodes['paper'].data['general_states']
        # general_states = paper_general_states
        shuf_general = batch_sub_g.nodes['paper'].data['shuf_general_states']
        specific_pair = {}
        # get shuf neg specific
        shuf_specific = {}
        for sec_type in ['intro', 'method', 'experiment', 'relate']:
            specific_s = batch_sub_g.nodes['paper'].data[sec_type+'_states'][batch_idx_s]
            specific_t = batch_sub_g.nodes['paper'].data[sec_type+'_states'][batch_idx_t]
            specific_pair[sec_type] = self.section_fc[sec_type](torch.cat([specific_s, specific_t], dim=-1))
            # shuf_specific[sec_type] = paper_specific_states[sec_type][batch_shuf_paper]
            shuf_specific[sec_type] = batch_sub_g.nodes['paper'].data['shuf_'+sec_type+'_states']

        mutual_loss = None
        if self.use_mutual:
            mutual_loss = self.mutual_info(batch_sub_g, general_states, paper_specific_states,
                                       shuf_general, shuf_specific)

        return batch_sub_g, general_pair, specific_pair, mutual_loss