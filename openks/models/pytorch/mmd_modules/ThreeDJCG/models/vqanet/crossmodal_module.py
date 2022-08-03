import torch
import torch.nn as nn
import torch.nn.functional as F
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.transformer.attention import MultiHeadAttention
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.transformer.utils import PositionWiseFeedForward
import random


class CrossAttnModule(nn.Module):  # cross-self-cross-self-cross...; the depth
    def __init__(self, hidden_size, head, depth):
        super().__init__()
        self.depth = depth
        self.A2A_Attn = nn.ModuleList()
        self.A2B_Attn = nn.ModuleList()
        self.B2A_Attn = nn.ModuleList()
        self.B2B_Attn = nn.ModuleList()
        for i in range(depth):
            self.A2B_Attn.append(MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head))
            self.B2A_Attn.append(MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head))
        self.A2A_Attn.append(nn.Identity())
        self.B2B_Attn.append(nn.Identity())
        for i in range(depth-1):
            self.A2A_Attn.append(MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head))
            self.B2B_Attn.append(MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head))

    def forward(self, A_feat, B_feat, A_mask=None, B_mask=None, A2A_weight=None, A2B_weight=None, B2A_weight=None, B2B_weight=None, attention_weight_way='add', depth=None):  # depth: the weight
        if depth is None:
            depth = [i for i in range(self.depth)]
        for i in depth:
            # queries, keys, values, attention_mask=None, attention_weights=None, way='mul'
            if i != 0:
                TMP_A_feat = self.A2A_Attn[i](A_feat, A_feat, A_feat, A_mask, A2A_weight, way=attention_weight_way)
                TMP_B_feat = self.B2B_Attn[i](B_feat, B_feat, B_feat, B_mask, B2B_weight, way=attention_weight_way)
            else:
                TMP_A_feat, TMP_B_feat = A_feat, B_feat
            B_feat = self.A2B_Attn[i](B_feat, A_feat, A_feat, A_mask, A2B_weight, way=attention_weight_way)
            A_feat = self.B2A_Attn[i](A_feat, B_feat, B_feat, B_mask, B2A_weight, way=attention_weight_way)
        return A_feat, B_feat


class CrossmodalModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, lang_num_size=300, det_channel=128, head=4):
        super().__init__()
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size

        self.object_fc = nn.Linear(det_channel, hidden_size)

        self.match = nn.Sequential(  # 4 related object type
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, 4, 1)
        )

        self.CrossAttention = CrossAttnModule(hidden_size, head, depth=2)

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1

        dist_weights = data_dict['dist_weights']
        attention_matrix_way = data_dict['attention_matrix_way']
        features = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size

        batch_size, num_proposal = features.shape[:2]
        lang_num_max = data_dict["vqa_question_embedding"].shape[1]
        #objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        data_dict["random"] = random.random()

        # generate the object-level feature
        feature0 = features.clone()
        # copy paste
        # if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
        #     obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
        #     obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
        #     for i in range(batch_size):
        #         obj_mask = torch.where(obj_masks[i, :] == True)[0]
        #         obj_len = obj_mask.shape[0]
        #         obj_lens[i] = obj_len

        #     obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
        #     obj_features = features.reshape(batch_size*num_proposal, -1)
        #     obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
        #     total_len = obj_mask.shape[0]
        #     obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
        #     j = 0
        #     for i in range(batch_size):
        #         obj_mask = torch.where(obj_masks[i, :] == False)[0]
        #         obj_len = obj_mask.shape[0]
        #         j += obj_lens[i]
        #         if obj_len < total_len - obj_lens[i]:
        #             feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
        #         else:
        #             feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

        object_fea = feature0[:, None, :, :].repeat(1, lang_num_max, 1, 1).reshape(batch_size*lang_num_max, num_proposal, -1)
        object_fea = self.object_fc(object_fea)
        lang_fea = data_dict["vqa_question_lang_fea"]
        dist_weights = data_dict['dist_weights'][:, None, :, :, :].repeat(1, lang_num_max, 1, 1, 1).reshape(batch_size*lang_num_max, -1, num_proposal, num_proposal)
        # print("features", features.shape, lang_fea.shape)

        # todo simplify
        # cross-attention
        updated_object_fea, updated_lang_fea = self.CrossAttention(
            A_feat = object_fea, B_feat = lang_fea, B_mask=data_dict["vqa_question_attention_mask"],
            A2A_weight = dist_weights, attention_weight_way = data_dict['attention_matrix_way'])

        # updated_object_fea = self.t2o_cross_attn(object_fea, lang_fea, lang_fea, data_dict["vqa_question_attention_mask"])
        # updated_lang_fea = self.o2t_cross_attn(lang_fea, object_fea, object_fea)

        related_object_confidence = self.match(updated_object_fea.permute(0,2,1)).permute(0,2,1)  # batch_size, num_proposals
        related_object_confidence = related_object_confidence.reshape(batch_size, lang_num_max, num_proposal, -1)
        data_dict["vqa_pred_related_object_confidence"] = related_object_confidence

        data_dict["updated_lang_fea"] = updated_lang_fea
        return data_dict

