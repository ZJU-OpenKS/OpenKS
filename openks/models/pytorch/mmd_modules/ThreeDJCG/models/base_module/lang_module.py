import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False,
                 emb_size=300, hidden_size=256):
        super().__init__()

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

        self.fc = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=.1)
        self.layer_norm = nn.LayerNorm(128)
        # self.mhatt = MultiHeadAttention(d_model=128, d_k=16, d_v=16, h=4, dropout=.1, identity_map_reordering=False,
        #                                 attention_module=None,
        #                                 attention_module_kwargs=None)

    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)
        lang_len = data_dict["lang_len_list"]
        #word_embs = data_dict["main_lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)
        #lang_len = data_dict["main_lang_len_list"]
        batch_size, len_nun_max, max_des_len = word_embs.shape[:3]

        word_embs = word_embs.reshape(batch_size * len_nun_max, max_des_len, -1)
        lang_len = lang_len.reshape(batch_size * len_nun_max)
        first_obj = data_dict["first_obj_list"].reshape(batch_size * len_nun_max)

        # masking
        if data_dict["istrain"][0] == 1 and random.random() < 0.5:
            for i in range(word_embs.shape[0]):
                word_embs[i, first_obj] = data_dict["unk"][0]
                len = lang_len[i]
                for j in range(int(len/5)):
                    num = random.randint(0, len-1)
                    word_embs[i, num] = data_dict["unk"][0]
        elif data_dict["istrain"][0] == 1:
            for i in range(word_embs.shape[0]):
                len = lang_len[i]
                for j in range(int(len/5)):
                    num = random.randint(0, len-1)
                    word_embs[i, num] = data_dict["unk"][0]

        # Reverse
        main_lang_len = data_dict["main_lang_len_list"]
        main_lang_len = main_lang_len.reshape(batch_size * len_nun_max)

        if data_dict["istrain"][0] == 1 and random.random() < 0.5:
            for i in range(word_embs.shape[0]):
                new_word_emb = copy.deepcopy(word_embs[i])
                new_len = lang_len[i] - main_lang_len[i]
                new_word_emb[:new_len] = word_embs[i, main_lang_len[i]:lang_len[i]]
                new_word_emb[new_len:lang_len[i]] = word_embs[i, :main_lang_len[i]]
                word_embs[i] = new_word_emb

        # lang_feat = pack_padded_sequence(word_embs, lang_len, batch_first=True, enforce_sorted=False)
        lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)

        out, lang_last = self.gru(lang_feat)

        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        if self.use_bidir:
            cap_emb = (cap_emb[:, :, :int(cap_emb.shape[2] / 2)] + cap_emb[:, :, int(cap_emb.shape[2] / 2):]) / 2

        b_s, seq_len = cap_emb.shape[:2]
        mask_queries = torch.ones((b_s, seq_len), dtype=torch.int)
        for i in range(b_s):
            mask_queries[i, cap_len[i]:] = 0
        attention_mask = (mask_queries == 0).unsqueeze(1).unsqueeze(1).cuda()  # (b_s, 1, 1, seq_len)
        data_dict["attention_mask"] = attention_mask

        lang_fea = F.relu(self.fc(cap_emb))  # batch_size, n, hidden_size
        lang_fea = self.dropout(lang_fea)
        lang_fea = self.layer_norm(lang_fea)
        # lang_fea = self.mhatt(lang_fea, lang_fea, lang_fea, attention_mask)

        data_dict["lang_fea"] = lang_fea

        # data_dict["lang_fea"] = cap_emb
        # print("lang_fea", lang_fea.shape)

        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # batch_size, hidden_size * num_dir
        # store the encoded language features
        data_dict["lang_emb"] = lang_last  # B, hidden_size
        # print("lang_last", lang_last.shape)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

