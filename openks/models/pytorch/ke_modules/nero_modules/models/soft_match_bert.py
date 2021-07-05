import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import os

from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizer

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)


    def softmax_mask(self, val, mask):
        rank_val = len(list(val.shape))
        rank_mask = len(list(mask.shape))
        if rank_val - rank_mask == 1:
            mask = torch.unsqueeze(mask, axis=-1)
        return  (0 - 1e30) * (1 - mask.float()) + val

    def forward(self, inputs, mask=None, keep_prob=1.0, is_train=True):
        x = torch.dropout(inputs, keep_prob, is_train)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        if mask is not None:
            x = self.softmax_mask(x, mask)
        x = F.softmax(x, dim=1)
        x = x.squeeze(-1)
        return x

class SoftMatch(nn.Module):
    def __init__(self, config, word_mat, word2idx_dict):
        super(SoftMatch, self).__init__()
        self.config = config

        self.bert = BertModel.from_pretrained('/home/ps/disk_sdb/yyr/codes/NEROtorch/pretrain_models/bert')
        self.bert_no_grad = BertModel.from_pretrained('/home/ps/disk_sdb/yyr/codes/NEROtorch/pretrain_models/bert')
        
        for name ,param in self.bert_no_grad.named_parameters():
            param.requires_grad = False


        self.hidden_size = self.config.hidden
        self.embedding_dim = self.config.glove_dim
        self.keep_prob = self.config.keep_prob
        self.is_train = True
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.array(word_mat)))
        self.embedding.weight.requires_grad = False

        self.attention_model = Attention(self.embedding_dim, self.hidden_size)

        self.fc_sent2rel = nn.Linear(768, self.config.num_class)
        self.fc_pat2rel = nn.Linear(768, self.config.num_class)


    def get_embedding(self, sent):

        return self.embedding(sent)


    def cosine(self, seq1, seq2):
        dim = seq1.shape[-1]
        norm1 = torch.norm(seq1 + 1e-5, dim=1, keepdim=True)
        norm2 = torch.norm(seq2 + 1e-5, dim=1, keepdim=True)
        sim = torch.matmul(seq1 / norm1, torch.transpose(seq2 / norm2, 1, 0))
        return sim

    def att_match(self, mid, pat, mid_mask, pat_mask, keep_prob=1.0, is_train=True):
        mid_d = torch.dropout(mid, keep_prob, is_train)
        pat_d = torch.dropout(pat, keep_prob, is_train)
        mid_a = self.attention_model(mid_d, mask=mid_mask, keep_prob=keep_prob, is_train=is_train)
        pat_a = self.attention_model(pat_d, mask=pat_mask, keep_prob=keep_prob, is_train=is_train)
        # import pdb; pdb.set_trace()
        mid_v = torch.sum(mid_a.unsqueeze(-1) * mid, dim=1)
        pat_v = torch.sum(pat_a.unsqueeze(-1) * pat, dim=1)
        pat_v_d = torch.sum(pat_a.unsqueeze(-1) * pat_d, dim=1)
        sur_sim = self.cosine(mid_v, pat_v_d)
        pat_sim = self.cosine(pat_v, pat_v_d)
        return sur_sim, pat_sim

    def mean_match(self, mid, pat, mid_mask, pat_mask, keep_prob, is_train=True):
        def mean(emb, mask):
            mask = mask.float()
            length = torch.sum(mask, dim=1)
            emb = torch.sum(emb, dim=1) / length.unsqueeze(1)
            return emb
        pat_d = torch.dropout(pat, keep_prob, is_train)
        mid_v = mean(mid, mid_mask)
        pat_v = mean(pat, pat_mask)
        pat_v_d = mean(pat_d, pat_mask)
        sur_sim = self.cosine(mid_v, pat_v_d)
        pat_sim = self.cosine(pat_v, pat_v_d)
        return sur_sim, pat_sim

    def forward(self, sent_tokens, pats_tokens, sent_tokens_mask, pats_tokens_mask, sent: torch.IntTensor, mid: torch.IntTensor, rel_label: torch.IntTensor,
                pat_label: torch.IntTensor, pattern_rels: torch.IntTensor, pats: torch.IntTensor, weights: torch.FloatTensor, is_train=True):
        """
        8是batchsize
        sent: 8 * 110, 现在还不是embedding
        mid: 8 * 110
        rel: 8 (每个元素小于rel_nums 3)
        pat: 8, (每个数字代表对应的pattern_id), 如果无对应的，就是-1


        patterns：
        pattern_rels: pattern_num * rel_num， 每条pattern对应的relation
        pats_token: pattern_num * pat_token_len， 每条patterns的embedding
        weights: pattern的权重, 维度: patterns_num
        """
        self.is_train = is_train
        device = self.config.device

        sent = torch.from_numpy(sent).long().to(device)
        mid = torch.from_numpy(mid).long().to(device)
        rel_label = torch.from_numpy(rel_label).long().to(device)
        pat_label = torch.from_numpy(pat_label).long().to(device)
        pattern_rels = torch.from_numpy(pattern_rels).float().to(device)
        pats = torch.from_numpy(pats).long().to(device)
        weights = torch.from_numpy(weights).float().to(device)
        sent_tokens = torch.from_numpy(sent_tokens).long().to(device)
        pats_tokens = torch.from_numpy(pats_tokens).long().to(device)
        sent_tokens_mask = torch.from_numpy(sent_tokens_mask).bool().to(device) 
        pats_tokens_mask = torch.from_numpy(pats_tokens_mask).bool().to(device)

        rel_label = torch.argmax(rel_label, -1)
        pattern_rels_label = torch.argmax(pattern_rels, -1)

        sent_mask = sent.bool()
        sent_len = torch.sum(sent_mask, dim=1)
        sent_max_len = torch.max(sent_len)
        sent_mask = sent_mask[:, :sent_max_len]
        sent = sent[:, :sent_max_len]

        mid_mask = mid.bool()
        mid_len = torch.sum(mid_mask, dim=1)
        mid_max_len = torch.max(mid_len)
        mid_mask = mid_mask[:, :mid_max_len]
        mid = mid[:, :mid_max_len]

        pat_mask = pats.bool()
        pat_len = torch.sum(pat_mask, dim=1)
        pat_max_len = torch.max(pat_len)
        pat_mask = pat_mask[:, :pat_max_len]
        pat = pats[:, :pat_max_len]

        sent_embedding = self.get_embedding(sent)
        mid_embedding = self.get_embedding(mid)
        pat_embedding = self.get_embedding(pats)

        # encoder

        sent_d = self.bert(sent_tokens, attention_mask=sent_tokens_mask)[0][:, 0, :]
        pat_d = self.bert_no_grad(pats_tokens, attention_mask=pats_tokens_mask)[0][:, 0, :]
        


        # similarity
        sim, pat_sim = self.att_match(mid_embedding, pat_embedding, mid_mask, pat_mask,
                                    self.keep_prob, self.is_train)

        neg_idxs = torch.matmul(pattern_rels, torch.transpose(pattern_rels, 1, 0))
        pat_pos = torch.square(torch.max(self.config.tau - pat_sim, torch.zeros_like(pat_sim)))
        pat_pos = torch.max(pat_pos - (1 - neg_idxs) * 1e30, dim=1)[0]
        pat_neg = torch.square(torch.max(pat_sim, torch.zeros_like(pat_sim)))
        pat_neg = torch.max(pat_neg - 1e30 * neg_idxs, dim=1)[0]
        l_sim = torch.sum(weights * (pat_pos + pat_neg), dim=0)



        logit = self.fc_sent2rel(sent_d)
        pred = F.softmax(logit, dim=1)

        if self.is_train is True:

            l_a = F.cross_entropy(logit[:self.config.gt_batch_size], rel_label[:self.config.gt_batch_size])

            xsim = sim[self.config.gt_batch_size:]
            # xsim = xsim.detach()
            # xsim.requires_grad = False
            pseudo_rel = pattern_rels_label[torch.argmax(xsim, dim=1)]
            bound = torch.max(xsim, dim=1)[0]
            weight = F.softmax(10 * bound, dim=0)

            l_u = torch.sum(weight * F.cross_entropy(logit[self.config.gt_batch_size:], pseudo_rel, reduction='none'))

            pat2rel = self.fc_pat2rel(pat_d)
            pat2rel_pred = F.softmax(pat2rel, dim=1)
            l_pat = F.cross_entropy(pat2rel_pred, pattern_rels_label)
            loss = l_a + self.config.alpha * l_pat + self.config.gamma * l_u + self.config.beta * l_sim
            # loss = l_a + self.config.alpha * l_pat + self.config.beta * l_u
        else:
            loss = 0.0

        preds = torch.argmax(pred, dim=1)
        val = torch.sum((0-torch.log(torch.clamp(pred, 1e-5, 1.0))) * pred, dim=1)
        golds = rel_label

        
        return golds, preds, val, loss



if __name__ == '__main__':
    batch_size = 4
    rel_num = 3
    pattern_num = 10

    sent_token_len = 20
    pat_token_len = 10

    sent = torch.randint(90, (batch_size, sent_token_len))
    mid = torch.randint(90, (batch_size, sent_token_len))

    rel_label = torch.LongTensor(batch_size).random_() % rel_num
    pat_label = torch.LongTensor(batch_size).random_() % pattern_num

    # pattern_rels = torch.LongTensor(pattern_num).random_() % rel_num
    pattern_rels = torch.zeros(pattern_num, rel_num).scatter_(1, torch.LongTensor(pattern_num, 1).random_() % rel_num, 1)

    pats = torch.randint(90, (pattern_num, pat_token_len))
    weights = torch.rand(pattern_num).abs_()

    model = SoftMatch()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    golds, preds, val, loss = model(sent, mid, rel_label, pat_label, pattern_rels, pats, weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
