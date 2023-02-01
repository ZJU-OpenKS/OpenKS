import torch
import torch.nn as nn
import dgl

class Discriminator(nn.Module):
    def __init__(self, device, hid_dim1, hid_dim2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(hid_dim1, hid_dim2, 1).to(device)
        self.act = nn.Sigmoid()
    def forward(self, c_x, pos_h, neg_h):
        sc_1 = torch.squeeze(self.f_k(c_x, pos_h), -1).sum(dim=-1)
        sc_1 = self.act(sc_1)
        sc_2 = torch.squeeze(self.f_k(
            c_x.unsqueeze(1).repeat_interleave(neg_h.size(1), dim=1), neg_h
        ), -1).sum(dim=-1)
        sc_2 = self.act(sc_2)
        return sc_1, sc_2.sum(1)


class MutualInfoMax(nn.Module):
    def __init__(self, device, section_fc, n_hid, alpha1, alpha2, alpha3):
        super(MutualInfoMax, self).__init__()
        self.device = device
        self.section_fc = section_fc
        self.n_hid = n_hid
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.disc = nn.ModuleDict().to(device)
        self.disc['general'] = Discriminator(device, n_hid//2, n_hid//2)
        self.disc['intro'] = Discriminator(device, n_hid//2, n_hid//2)
        self.disc['method'] = Discriminator(device, n_hid//2, n_hid//2)
        self.disc['experiment'] = Discriminator(device, n_hid//2, n_hid//2)
        self.disc['relate'] = Discriminator(device, n_hid*3//2, n_hid*3//2)
        # self.disc['edge'] = Discriminator_edge(device, n_hid, n_hid)

    def sp_func(self, arg):
        return torch.log(1+torch.exp(arg))

    def mi_loss_jsd(self, pos, neg):
        e_pos = self.sp_func(-pos)
        e_neg = self.sp_func(neg)
        return e_pos+e_neg

    def forward(self, graph_encodes, general_states, specific_states, shuf_general, shuf_specific):
        # dependent loss between general and each specific states
        paper_size = graph_encodes.num_nodes('paper')
        all_states = []
        # all states shape batch_size * 1 * 4 * h
        for sec_type in ['general', 'intro', 'method', 'experiment']:
            all_states.append(graph_encodes.nodes['paper'].data[sec_type+'_states'])
        all_states = torch.stack(all_states, dim=-2)
        # out shape batch_size * 1 * 4 * 4
        out = torch.matmul(all_states, all_states.transpose(2, 3))
        deno = torch.max(out, dim=-1)[0]
        deno.masked_fill_(deno==0, 1)
        out = out / deno.unsqueeze(-2)
        target = torch.eye(all_states.shape[-2], all_states.shape[-2]).unsqueeze(0).unsqueeze(1).to(self.device)
        graph_encodes.nodes['paper'].data['loss_diff'] = \
            torch.abs(out-target).view(paper_size, -1).sum(-1)
        loss_mutual = self.alpha1 * dgl.readout_nodes(graph_encodes, 'loss_diff', op='mean', ntype='paper')

        # mutual loss between target and readout(support graph) for general states
        readout_general = dgl.readout_nodes(graph_encodes, 'general_states', op='mean', ntype='paper')
        readout_general = dgl.broadcast_nodes(graph_encodes, readout_general, ntype='paper')
        general_pos, general_neg = self.disc['general'](readout_general, general_states, shuf_general)
        graph_encodes.nodes['paper'].data['loss_mutual_general'] = self.mi_loss_jsd(general_pos, general_neg)
        loss_mutual += self.alpha2 * dgl.readout_nodes(graph_encodes, 'loss_mutual_general', op='mean', ntype='paper')

        # mutual loss between target and neighbors for specific states
        for sec_type in ['intro', 'method', 'experiment', 'relate']:
            specific_pos, specific_neg = \
                self.disc[sec_type](specific_states[sec_type], graph_encodes.nodes['paper'].data[sec_type+'_states'],
                                    shuf_specific[sec_type])
            graph_encodes.nodes['paper'].data['loss_mutual_'+sec_type] = self.mi_loss_jsd(specific_pos, specific_neg)
            loss_mutual += self.alpha3 * dgl.readout_nodes(graph_encodes, 'loss_mutual_'+sec_type, op='mean', ntype='paper')

        # mutual loss for relate relation
        # for etype in graph_encodes.etypes:
        #     if 'citing' in etype:
        #         sec_type = etype.split('-')[1]
        #         src, dst = graph_encodes.edges(etype=etype)
        #         general_states = torch.cat((graph_encodes.nodes['paper'].data['general_states'][src],
        #                                     graph_encodes.nodes['paper'].data['general_states'][dst]), dim=-1)
        #         specific_states = torch.cat((graph_encodes.nodes['paper'].data[sec_type + '_states'][src],
        #                                      graph_encodes.nodes['paper'].data[sec_type + '_states'][dst]), dim=-1)
        #         general_states = self.section_fc['general'](general_states)
        #         specific_states = self.section_fc[sec_type](specific_states)
        #         states = torch.cat([general_states, specific_states], dim=-1)
        #         edge_pos, edge_neg = self.disc['edge'](states, edge_states[etype], shuf_edge[etype])
        #         graph_encodes.edges[etype].data['loss_mutual_edge'] = self.mi_loss_jsd(edge_pos, edge_neg)
        #         loss_mutual += self.alpha4 * dgl.readout_edges(graph_encodes, 'loss_mutual_edge', op='mean', etype=etype)
        return loss_mutual