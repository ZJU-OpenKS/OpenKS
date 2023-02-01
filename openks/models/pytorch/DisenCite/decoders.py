import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions import Attention
import time


class GRUDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, drop_emb, dropout=0.35):
        super(GRUDecoder, self).__init__()
        # self.max_seq_len = max_seq_len
        # self.lstm = nn.LSTMCell(in_dim, out_dim)
        self.drop_emb = drop_emb
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(in_dim, out_dim, num_layers=n_layers, batch_first=True, bidirectional=False, dropout=dropout)
        # self.attn_model = attn_model
        # self.is_pointer_gen = is_pointer_gen

        # p_vocab
        # self.out1 = nn.Linear(out_dim*2, out_dim)
        # self.out2 = nn.Linear(out_dim, vocab_size)

    def forward(self, inp_emb, hiddens_decoder):
        # for i in range(self.max_seq_len):
        # inp_emb = inp_embs[:, i, :]
        if self.drop_emb:
            inp_emb = self.dropout(inp_emb)
        lstm_out, hiddens_decoder = self.gru(inp_emb, hiddens_decoder)
        # hiddens_decoder = (h_t, c_t)

        return lstm_out, hiddens_decoder


class DisenPaperDecoder(nn.Module):
    def __init__(self, device, section_fc, emb_vocab, vocab, args):
        super(DisenPaperDecoder, self).__init__()
        self.section_fc = section_fc
        self.max_len_citation = args.n_max_len_citation
        self.is_dec_attention = args.is_dec_attention
        self.is_pointer_gen = args.is_pointer_gen
        self.is_coverage = args.is_coverage
        self.n_hid = args.n_hid
        self.vocab = vocab
        self.emb_vocab = emb_vocab
        self.drop = nn.Dropout(args.dropout)
        vocab_size = emb_vocab.weight.size(0)
        self.text_decoder = GRUDecoder(args.n_hid, args.n_hid, args.n_dec_layer, args.is_drop_emb).to(device)

        if args.is_pointer_gen:
            self.p_gen_linear = nn.Linear(args.n_hid * 4, 3).to(device)
        if self.is_dec_attention:
            self.attn_model = Attention(device, args.n_hid, args.is_coverage).to(device)
            self.x_context = nn.Linear(args.n_hid * 2, args.n_hid).to(device)
            self.out = nn.Sequential(nn.Linear(args.n_hid * 2, args.n_hid), nn.Linear(args.n_hid, vocab_size)).to(device)
        else:
            self.out = nn.Sequential(nn.Linear(args.n_hid, args.n_hid), nn.Linear(args.n_hid, vocab_size)).to(device)

    def forward(self, src_outs, dst_outs, y_t_1, s_t_1, c_t_1,
                enc_extend_vocab, enc_padding_mask, extra_zeros, coverage, step):
        if self.is_dec_attention:
            if not self.training and step == 0:
                # c_t_src, c_t_dst, _, coverage_next = \
                #     self.attn_model(src_outs, dst_outs, enc_padding_mask, s_t_1[-1], coverage)

                # c_t_src, _, coverage_next_src = \
                #     self.attn_model(src_outs, enc_padding_mask[:, 0, :], s_t_1[-1], coverage[:, 0, :])
                # c_t_dst, _, coverage_next_dst = \
                #     self.attn_model(dst_outs, enc_padding_mask[:, 1, :], s_t_1[-1], coverage[:, 1, :])
                enc_outs = torch.cat([src_outs, dst_outs], dim=1)
                c_t, _, coverage_next = self.attn_model(enc_outs, enc_padding_mask, s_t_1[-1], coverage)
                # coverage_next = torch.stack([coverage_next_src, coverage_next_dst], dim=1)
                coverage = coverage_next

            y_t_1_emb = self.emb_vocab(y_t_1)
            x = self.x_context(torch.cat([c_t_1, y_t_1_emb], dim=-1))
            # x = y_t_1_emb
            out, s_t = self.text_decoder(x.unsqueeze(1), s_t_1)
            # c_t_src, c_t_dst, attn_dist, coverage_next = \
            #     self.attn_model(src_outs, dst_outs, enc_padding_mask, s_t[-1], coverage)
            # c_t_src, attn_dist_src, coverage_next_src = \
            #     self.attn_model(src_outs, enc_padding_mask[:, 0, :], s_t[-1], coverage[:, 0, :])
            # c_t_dst, attn_dist_dst, coverage_next_dst = \
            #     self.attn_model(dst_outs, enc_padding_mask[:, 1, :], s_t[-1], coverage[:, 1, :])
            # coverage_next = torch.stack([coverage_next_src, coverage_next_dst], dim=1)
            # attn_dist = torch.stack([attn_dist_src, attn_dist_dst], dim=1)

            enc_outs = torch.cat([src_outs, dst_outs], dim=1)
            c_t, attn_dist, coverage_next = self.attn_model(enc_outs, enc_padding_mask, s_t_1[-1], coverage)

            if self.training or step > 0:
                coverage = coverage_next

            output = torch.cat([out.squeeze(1), c_t], dim=-1)
            # output = out.squeeze(1)
            output = self.out(output)
            vocab_dist = F.softmax(output, dim=-1)
            p_gen = None
            if self.is_pointer_gen:
                p_gen_input = torch.cat([c_t, s_t[-1], x], dim=-1)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.softmax(p_gen, dim=-1)
                vocab_dist = p_gen[:, 0].unsqueeze(dim=-1) * vocab_dist
                attn_dist = p_gen[:, 1:].unsqueeze(dim=-1) * attn_dist
                if extra_zeros is not None:
                    vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
                batch_size = src_outs.size(0)
                # final_dist = vocab_dist.scatter_add(1, enc_extend_vocab.view(batch_size, -1),
                #                                     attn_dist.view(batch_size, -1))
                final_dist = vocab_dist.scatter_add(1, enc_extend_vocab.view(batch_size, -1),
                                                    torch.cat([attn_dist[:, 0, :], attn_dist[:, 1, :]], dim=-1))
            else:
                final_dist = vocab_dist
        else:
            y_t_1_emb = self.emb_vocab(y_t_1)
            x = y_t_1_emb.unsqueeze(1)
            out, s_t = self.text_decoder(x, s_t_1)
            output = self.out(out.squeeze(1))
            final_dist = F.softmax(output, dim=-1)
            c_t = c_t_1
            # c_t_dst = c_t_1_dst
            attn_dist = None
            p_gen = None
            coverage = coverage
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

    # def forward(self, y_t_1, s_t_1, step=None):
    #     y_t_1_emb = self.emb_vocab(y_t_1)
    #     out, s_t = self.text_decoder(y_t_1_emb, s_t_1)
    #     # output = self.out(out.squeeze(1))
    #     output = self.out(out[:, -1, :])
    #     # print(output.shape)
    #
    #     final_dist = F.log_softmax(output, dim=1)
    #     return final_dist, s_t

    def beam_search(self, beam_scorer, src_outs, dst_outs, s_t_1, c_t_1,
                    batch_enc_extend_vocab, enc_padding_mask, extra_zeros, coverage):
        device = src_outs.device
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        # input_ids: (num_beams * batch_size, cur_len)
        input_ids = torch.ones(num_beams * batch_size, 1, dtype=torch.long, device=device) * self.vocab['SOS']
        batch_beam_size, cur_len = input_ids.shape
        assert (
                num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # src_outs/dst_outs: (num_beams * batch_size, src_len, n_hid)
        src_outs = src_outs.repeat_interleave(num_beams, dim=0)
        dst_outs = dst_outs.repeat_interleave(num_beams, dim=0)
        # s_t_1: (src_len, num_beams * batch_size, n_hid)
        s_t_1 = s_t_1.repeat_interleave(num_beams, dim=1)
        # enc_padding_mask: (num_beams * batch_size, 2, src_len)
        enc_padding_mask = enc_padding_mask.repeat_interleave(num_beams, dim=0)
        # extra_zeros: (num_beams * batch_size, num_oov)
        if extra_zeros is not None:
            extra_zeros = extra_zeros.repeat_interleave(num_beams, dim=0)
        # c_t_1_src/c_t_1_dst: (num_beams * batch_size, n_hid)
        c_t_1 = c_t_1.repeat_interleave(num_beams, dim=0)
        # c_t_1_dst = c_t_1_dst.repeat_interleave(num_beams, dim=0)
        coverage = coverage.repeat_interleave(num_beams, dim=0)

        while cur_len < self.max_len_citation:
            cur_input = input_ids[:, -1]
            cur_input = torch.where(cur_input < len(self.vocab), cur_input,
                                    torch.LongTensor([self.vocab['UNK']]).to(device))
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self(
                src_outs, dst_outs, cur_input, s_t_1, c_t_1,
                batch_enc_extend_vocab, enc_padding_mask, extra_zeros, coverage, cur_len-1
            )
            # next_token_scores: (num_beams * batch_size, vocab_size)
            next_token_scores = torch.log(final_dist + 1e-12)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids, next_token_scores,
                next_tokens, next_indices,
                pad_token_id=self.vocab['PAD'],
                eos_token_id=self.vocab['EOS']
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len += 1

            if beam_scorer.is_done:
                break

        # decoded: (1, gen_len)
        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices,
            pad_token_id=self.vocab['PAD'], eos_token_id=self.vocab['EOS']
        )
        return decoded['sequences']
