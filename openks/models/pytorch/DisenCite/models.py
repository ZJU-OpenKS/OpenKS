import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import DisenPaperEncoder
from decoders import DisenPaperDecoder
from generation_utils import BeamSearchScorer
from attentions_pos import Attention
import time


class CitationGeneration(nn.Module):
    def __init__(self, device, heter_graph, section_type2index, vocab2index, vec, args):
        super(CitationGeneration, self).__init__()
        self.device = device
        self.max_len_section = args.n_max_len_section
        self.max_len_citation = args.n_max_len_citation
        self.n_decoder_layers = args.n_dec_layer
        self.n_hid = args.n_hid
        self.is_coverage = args.is_coverage
        self.cov_loss_wt = args.cov_loss_wt
        self.vocab2index = vocab2index
        self.vocab_size = args.vocab_size
        # self.bow_smoothing = args.bow_smoothing
        self.n_beams = args.n_beams
        self.emb_vocab = nn.Embedding(args.vocab_size, args.n_hid, padding_idx=vocab2index['PAD'])
        if not args.pretrained:
            self.emb_vocab.from_pretrained(vec, freeze=False)
        section_fc = nn.ModuleDict().to(device)
        for sec_type in ['general', 'intro', 'method', 'experiment']:
            section_fc.update({sec_type: nn.Linear(args.n_hid, args.n_hid // 2)})
        section_fc.update({'relate': nn.Linear(args.n_hid * 3, args.n_hid // 2)})
        self.encoder = DisenPaperEncoder(device, heter_graph, section_type2index, section_fc,
                                         self.emb_vocab, args).to(device)
        self.decoder = DisenPaperDecoder(device, section_fc, self.emb_vocab, vocab2index, args).to(device)
        self.attn_model = Attention(device, self.n_hid, self.is_coverage).to(device)

        # self.pos_pre = nn.Linear(args.n_hid // 2, 4).to(device)
        self.pos_pre = nn.Linear(args.n_hid//2*7, 4).to(device)
        self.pos_criterion = nn.BCELoss(reduction='none')
        self.use_mutual = args.use_mutual

    def forward(self, batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos,
                batch_shuf_paper, batch_enc_extend_vocab,
                batch_citation_inp, batch_citation_target,
                enc_padding_mask, dec_padding_mask, batch_citation_len, citation_len_max, extra_zeros):
        batch_size = batch_sub_g.batch_size
        graph_encodes, general_pair, specific_pair, mutual_loss = \
            self.encoder(batch_sub_g, batch_pair, batch_shuf_paper)
        # pos_score = self.pos_pre(general_pair[:, -1, :])
        # pos_score = torch.sigmoid(pos_score)

        # batch_size, 4, num_layer, n_hid/2
        specific_pair_h = torch.stack([specific_pair['intro'], specific_pair['method'],
                                specific_pair['experiment'], specific_pair['relate']], dim=1)

        cont = specific_pair_h[torch.arange(batch_size), batch_current_pos, :, :]

        s_t_1 = torch.cat([general_pair, cont], dim=-1).transpose(0, 1)

        # bow_preds = F.softmax(self.bow_pre(cont[:, -1, :]), dim=-1)
        # smoothed_bow = batch_citation_bow * (1-self.bow_smoothing) + self.bow_smoothing/self.vocab_size
        # bow_loss = self.bow_criterion(bow_preds, smoothed_bow).mean(dim=-1)

        # s_t_1_sp: (batch_size, 4, num_layer, n_hid / 2)
        # s_t_1_sp = torch.stack([specific_pair['intro'], specific_pair['method'],
        #                         specific_pair['experiment'], specific_pair['relate']], dim=1)
        # s_t_1_sp = torch.cat([specific_pair['intro'], specific_pair['method'],
        #                         specific_pair['experiment']], dim=-1)
        # s_t_1_sp: (batch_size, num_layer, n_hid / 2)
        # s_t_1_sp = torch.sum(pos_score.unsqueeze(-1).unsqueeze(-1) * s_t_1_sp, dim=1)
        # s_t_1: (num_layer, batch_size, n_hid)

        # s_t_1 = torch.cat([general_pair, s_t_1_sp], dim=-1).transpose(0, 1)

        c_t_1 = torch.zeros([batch_size, self.n_hid]).to(self.device)
        # c_t_1_dst = torch.zeros([batch_size, self.n_hid]).to(self.device)

        src, dst = batch_pair[:, 0], batch_pair[:, 1]
        src_seq_outs = []
        dst_seq_outs = []
        for sec_type in ['intro', 'method', 'experiment']:
            src_seq_outs.append(graph_encodes.nodes['paper'].data[sec_type+'_seq_states'][src])
            dst_seq_outs.append(graph_encodes.nodes['paper'].data[sec_type+'_seq_states'][dst])
        # batch_size * (3*seq_len) * h
        src_outs = torch.cat(src_seq_outs, dim=1)
        dst_outs = torch.cat(dst_seq_outs, dim=1)
        # batch_size * 3 * seq_len * h
        # src_outs = torch.stack(src_seq_outs, dim=1)
        # dst_outs = torch.stack(dst_seq_outs, dim=1)

        encode_pair = torch.cat([general_pair, specific_pair['intro'], specific_pair['method'],
                                 specific_pair['experiment'], specific_pair['relate']], dim=-1)
        enc_outs = torch.cat([src_outs, dst_outs], dim=1)
        c_t, attn_dist, coverage_next = self.attn_model(enc_outs, enc_padding_mask, encode_pair[:, -1, :], None)

        pos_score = torch.sigmoid(self.pos_pre(torch.cat([encode_pair[:, -1, :], c_t], dim=-1)))
        # pos_score = torch.sigmoid(self.pos_pre(encode_pair[:, -1, :]))

        pos_loss = self.pos_criterion(pos_score, batch_citation_pos).mean(dim=-1)

        # batch_size * 2 * (3*seq_len)
        coverage = torch.zeros_like(batch_enc_extend_vocab, dtype=torch.float).to(self.device)
        step_losses = []
        for di in range(min(self.max_len_citation, citation_len_max)):
            y_t_1 = batch_citation_inp[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                self.decoder(src_outs, dst_outs, y_t_1, s_t_1, c_t_1,
                             batch_enc_extend_vocab, enc_padding_mask,
                             extra_zeros, coverage, di)
            target = batch_citation_target[:, di].unsqueeze(1)
            gold_probs = torch.gather(final_dist, 1, target).squeeze(1)
            step_loss = -torch.log(gold_probs + 1e-12)
            # if torch.mean(step_loss).item() != torch.mean(step_loss).item():
            #     print(step_loss)
            #     print(gold_probs)
            #     print(final_dist)
            #     print(target[-1])
            #     print(final_dist[-1])

            if self.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage).view(batch_size, -1), 1)
                step_loss = step_loss + self.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / citation_len_max
        if self.use_mutual:
            loss = torch.mean(pos_loss + batch_avg_loss + mutual_loss, dim=0)
        else:
            loss = torch.mean(pos_loss + batch_avg_loss, dim=0)
        # loss = torch.mean(batch_avg_loss, dim=0)

        # print(loss.item())
        # if loss.item() != loss.item():
        #     print(batch_avg_loss)
        #     print(torch.mean(batch_avg_loss, dim=0))
        #     print(torch.mean(bow_loss, dim=0))
        #     print(torch.mean(mutual_loss, dim=0))
        #     print(torch.mean(pos_loss, dim=0))
        #     time.sleep(100)
        # loss = torch.mean(batch_avg_loss + mutual_loss, dim=0)
        # loss = torch.mean(batch_avg_loss, dim=0)
        return loss

    def beam_search(self, batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos, batch_shuf_paper,
                    batch_enc_extend_vocab, enc_padding_mask, extra_zeros):
        batch_size = batch_sub_g.batch_size
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size, max_length=self.max_len_citation,
            num_beams=self.n_beams, device=self.device,
        )

        graph_encodes, general_pair, specific_pair, mutual_loss = \
            self.encoder(batch_sub_g, batch_pair, batch_shuf_paper)

        # pos_score = self.pos_pre(general_pair[:, -1, :])
        # pos_score = torch.sigmoid(pos_score)

        # encode_pair = torch.cat([general_pair, specific_pair['intro'], specific_pair['method'],
        #                             specific_pair['experiment'], specific_pair['relate']], dim=-1)
        # pos_score = torch.sigmoid(self.pos_pre(encode_pair[:, -1, :]))

        # pos_score = self.pos_pre(general_pair[:, -1, :])

        # s_t_1_sp = torch.stack([specific_pair['intro'], specific_pair['method'],
        #                         specific_pair['experiment'], specific_pair['relate']], dim=1)
        # s_t_1_sp = torch.sum(pos_score.unsqueeze(-1).unsqueeze(-1) * s_t_1_sp, dim=1)
        # s_t_1_sp = torch.cat([specific_pair['intro'], specific_pair['method'],
        #                         specific_pair['experiment']], dim=-1)
        # s_t_1 = torch.cat([general_pair, s_t_1_sp], dim=-1).transpose(0, 1)
        specific_pair_h = torch.stack([specific_pair['intro'], specific_pair['method'],
                                       specific_pair['experiment'], specific_pair['relate']], dim=1)
        cont = specific_pair_h[torch.arange(batch_size), batch_current_pos, :, :]
        s_t_1 = torch.cat([general_pair, cont], dim=-1).transpose(0, 1)

        c_t_1 = torch.zeros([batch_pair.size(0), self.n_hid]).to(self.device)
        # c_t_1_dst = torch.zeros([batch_pair.size(0), self.n_hid]).to(self.device)

        src, dst = batch_pair[:, 0], batch_pair[:, 1]
        src_seq_outs = []
        dst_seq_outs = []
        for sec_type in ['intro', 'method', 'experiment']:
            src_seq_outs.append(graph_encodes.nodes['paper'].data[sec_type + '_seq_states'][src])
            dst_seq_outs.append(graph_encodes.nodes['paper'].data[sec_type + '_seq_states'][dst])

        src_outs = torch.cat(src_seq_outs, dim=1)
        dst_outs = torch.cat(dst_seq_outs, dim=1)

        encode_pair = torch.cat([general_pair, specific_pair['intro'], specific_pair['method'],
                                 specific_pair['experiment'], specific_pair['relate']], dim=-1)
        enc_outs = torch.cat([src_outs, dst_outs], dim=1)
        # print(enc_outs.shape)
        # print(encode_pair[:, -1, :].shape)
        c_t, attn_dist, coverage_next = self.attn_model(enc_outs, enc_padding_mask, encode_pair[:, -1, :], None)

        pos_score = torch.sigmoid(self.pos_pre(torch.cat([encode_pair[:, -1, :], c_t], dim=-1)))

        # pos_score = torch.sigmoid(self.pos_pre(encode_pair[:, -1, :]))

        # src_outs = torch.stack(src_seq_outs, dim=1)
        # dst_outs = torch.stack(dst_seq_outs, dim=1)
        coverage = torch.zeros_like(batch_enc_extend_vocab, dtype=torch.float).to(self.device)
        return self.decoder.beam_search(beam_scorer, src_outs, dst_outs, s_t_1, c_t_1,
                                        batch_enc_extend_vocab, enc_padding_mask, extra_zeros, coverage), pos_score

    def save(self, path):
        torch.save({
            'model_state': self.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
