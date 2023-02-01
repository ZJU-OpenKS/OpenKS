import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import time

class Attention(nn.Module):
    def __init__(self, device, n_hid, is_converage):
        super(Attention, self).__init__()
        self.n_hid = n_hid
        self.dec_proj = nn.Linear(n_hid, n_hid).to(device)
        self.enc_proj = nn.Linear(n_hid, n_hid).to(device)
        if is_converage:
            self.W_c = nn.Linear(1, n_hid, bias=False)
        self.is_coverage = is_converage
        self.v = nn.Linear(n_hid, 1, bias=False)

    def forward(self, enc_outs, enc_padding_mask, s_t_hat, coverage):
        batch_size = enc_outs.size(0)
        enc_features = self.enc_proj(enc_outs)
        dec_fea = self.dec_proj(s_t_hat)
        dec_fea_expanded = dec_fea.view(batch_size, -1, self.n_hid)
        # batch_size * seq_len * h
        att_features = enc_features + dec_fea_expanded
        if self.is_coverage:
            coverage = coverage.unsqueeze(dim=-1)
            coverage_feature = self.W_c(coverage)
            att_features = att_features + coverage_feature
        e = torch.tanh(att_features)
        scores = self.v(e)
        scores = scores.view(batch_size, -1)
        attn_dist_ = F.softmax(scores, dim=-1) * enc_padding_mask.view(batch_size, -1)
        normalization_factor = attn_dist_.sum(dim=-1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        # batch_size * 1 * seq_len
        attn_dist = attn_dist.unsqueeze(dim=1)
        enc_outs = enc_outs.view(batch_size, -1, self.n_hid)

        c_t = torch.bmm(attn_dist, enc_outs).squeeze(dim=1)

        attn_dist = attn_dist.view(batch_size, -1)

        if self.is_coverage:
            coverage = coverage.squeeze(dim=-1)
            coverage = coverage + attn_dist
        return c_t, attn_dist, coverage

# class CrossAttention(nn.Module):
#     def __init__(self, device, section_fc, n_hid, is_converage):
#         super(CrossAttention, self).__init__()
#         self.section_fc = section_fc
#         self.n_hid = n_hid
#         self.dec_proj = nn.Linear(n_hid, n_hid).to(device)
#         self.enc_proj = nn.Linear(n_hid, n_hid).to(device)
#         if is_converage:
#             self.W_c = nn.Linear(1, n_hid, bias=False)
#         self.is_coverage = is_converage
#         self.v = nn.Linear(n_hid, 1, bias=False)
#
#     def forward(self, src_outs, dst_outs, enc_padding_mask, s_t_hat, coverage):
#         batch_size = src_outs.size(0)
#
#         src_features = self.enc_proj(src_outs)
#         dst_features = self.enc_proj(dst_outs)
#
#         dec_fea = self.dec_proj(s_t_hat)
#         dec_fea_expanded = dec_fea.view(batch_size, -1, self.n_hid)
#         att_features_src = src_features + dec_fea_expanded
#         att_features_dst = dst_features + dec_fea_expanded
#         if self.is_coverage:
#             coverage = coverage.unsqueeze(dim=-1)
#             coverage_feature_src = self.W_c(coverage[:, 0, :, :])
#             coverage_feature_dst = self.W_c(coverage[:, 1, :, :])
#             att_features_src = att_features_src + coverage_feature_src
#             att_features_dst = att_features_dst + coverage_feature_dst
#         e_src = torch.tanh(att_features_src)
#         e_dst = torch.tanh(att_features_dst)
#         scores_src = self.v(e_src)
#         scores_dst = self.v(e_dst)
#         scores_src = scores_src.view(batch_size, -1)
#         scores_dst = scores_dst.view(batch_size, -1)
#         attn_dist_src_ = F.softmax(scores_src, dim=1) * enc_padding_mask[:, 0, :].view(batch_size, -1)
#         attn_dist_dst_ = F.softmax(scores_dst, dim=1) * enc_padding_mask[:, 1, :].view(batch_size, -1)
#         normalization_factor_src = attn_dist_src_.sum(dim=1, keepdim=True)
#         normalization_factor_dst = attn_dist_dst_.sum(dim=1, keepdim=True)
#         attn_dist_src = attn_dist_src_ / normalization_factor_src
#         attn_dist_dst = attn_dist_dst_ / normalization_factor_dst
#         attn_dist_src = attn_dist_src.unsqueeze(dim=1)
#         attn_dist_dst = attn_dist_dst.unsqueeze(dim=1)
#         enc_outs_src = src_outs.view(batch_size, -1, self.n_hid)
#         enc_outs_dst = dst_outs.view(batch_size, -1, self.n_hid)
#
#         c_t_src = torch.bmm(attn_dist_src, enc_outs_src).squeeze(dim=1)
#         c_t_dst = torch.bmm(attn_dist_dst, enc_outs_dst).squeeze(dim=1)
#
#         attn_dist_src = attn_dist_src.view(batch_size, -1)
#         attn_dist_dst = attn_dist_dst.view(batch_size, -1)
#         attn_dist = torch.stack([attn_dist_src, attn_dist_dst], dim=1)
#
#         if self.is_coverage:
#             coverage = coverage.squeeze(dim=-1)
#             coverage = coverage + attn_dist
#         return c_t_src, c_t_dst, attn_dist, coverage

# class CrossAttention(nn.Module):
#     def __init__(self, device, section_fc, n_hid, is_converage):
#         super(CrossAttention, self).__init__()
#         self.section_fc = section_fc
#         self.n_hid = n_hid
#         self.dec_proj = nn.Linear(n_hid, n_hid).to(device)
#         self.enc_proj = nn.Linear(n_hid, n_hid).to(device)
#         if is_converage:
#             self.W_c = nn.Linear(1, n_hid, bias=False)
#         self.is_coverage = is_converage
#         self.v = nn.Linear(n_hid, 1, bias=False)
#
#     def forward(self, src_outs, dst_outs, enc_padding_mask, s_t_hat, coverage):
#         batch_size = src_outs.size(0)
#         # print(src_outs.shape)
#         src_features = self.enc_proj(src_outs)
#         dst_features = self.enc_proj(dst_outs)
#
#         dec_fea = self.dec_proj(s_t_hat)
#         dec_fea_expanded = dec_fea.view(batch_size, -1, self.n_hid)
#         att_features_src = src_features + dec_fea_expanded
#         att_features_dst = dst_features + dec_fea_expanded
#         if self.is_coverage:
#             coverage = coverage.unsqueeze(dim=-1)
#             coverage_feature_src = self.W_c(coverage[:, 0, :, :])
#             coverage_feature_dst = self.W_c(coverage[:, 1, :, :])
#             att_features_src = att_features_src + coverage_feature_src
#             att_features_dst = att_features_dst + coverage_feature_dst
#         e_src = torch.tanh(att_features_src)
#         e_dst = torch.tanh(att_features_dst)
#         scores_src = self.v(e_src)
#         scores_dst = self.v(e_dst)
#         scores_src = scores_src.view(batch_size, -1)
#         scores_dst = scores_dst.view(batch_size, -1)
#         scores = torch.cat([scores_src, scores_dst], dim=-1)
#         enc_padding_mask_src = enc_padding_mask[:, 0, :].view(batch_size, -1)
#         enc_padding_mask_dst = enc_padding_mask[:, 1, :].view(batch_size, -1)
#         enc_padding_mask = torch.cat([enc_padding_mask_src, enc_padding_mask_dst], dim=-1)
#         attn_dist_ = F.softmax(scores, dim=-1) * enc_padding_mask
#         # attn_dist_src_ = F.softmax(scores_src, dim=1) * enc_padding_mask[:, 0, :].view(batch_size, -1)
#         # attn_dist_dst_ = F.softmax(scores_dst, dim=1) * enc_padding_mask[:, 1, :].view(batch_size, -1)
#         # normalization_factor_src = attn_dist_src_.sum(dim=1, keepdim=True)
#         # normalization_factor_dst = attn_dist_dst_.sum(dim=1, keepdim=True)
#         normalization_factor = attn_dist_.sum(dim=-1, keepdim=True)
#         attn_dist = attn_dist_ / normalization_factor
#         # attn_dist_src = attn_dist_src_ / normalization_factor_src
#         # attn_dist_dst = attn_dist_dst_ / normalization_factor_dst
#         # attn_dist_src = attn_dist_src.unsqueeze(dim=1)
#         # attn_dist_dst = attn_dist_dst.unsqueeze(dim=1)
#         attn_dist = attn_dist.unsqueeze(dim=1)
#         enc_outs_src = src_outs.view(batch_size, -1, self.n_hid)
#         enc_outs_dst = dst_outs.view(batch_size, -1, self.n_hid)
#         enc_outs = torch.cat([enc_outs_src, enc_outs_dst], dim=1)
#         c_t = torch.bmm(attn_dist, enc_outs).squeeze(dim=1)
#         # c_t_src = torch.bmm(attn_dist_src, enc_outs_src).squeeze(dim=1)
#         # c_t_dst = torch.bmm(attn_dist_dst, enc_outs_dst).squeeze(dim=1)
#         # attn_dist_src = attn_dist_src.view(batch_size, -1)
#         # attn_dist_dst = attn_dist_dst.view(batch_size, -1)
#         # attn_dist = torch.stack([attn_dist_src, attn_dist_dst], dim=1)
#
#         if self.is_coverage:
#             coverage = coverage.squeeze(dim=-1)
#             coverage = coverage + attn_dist
#         return c_t, attn_dist, coverage
