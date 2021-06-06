import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils import spectral_norm
# from energy_model.roi_pooling import make_pooler
# from energy_model.energy_criterion import EnergyCriterion
# from lib.layers import Conv2d, FrozenBatchNorm2d, BatchNorm2d
# from util.box_ops import box_cxcywh_to_xyxy
# from models.act_enhance import MLP


class Discriminator(nn.Module):
    """
    A simple discriminator for single triplet matching score computing.
    In this version, only focus on verb logits.
    """
    def __init__(self, d_model=256, nhead=8, dropout=0, pooling_dim=128):
        super().__init__()
        # self.pos_embed1 = nn.Linear(8, 32)
        # # self.pos_bn = nn.BatchNorm1d(32, momentum=0.001)
        # self.pos_embed2 = nn.Sequential(*[
        #     nn.Linear(32, 128), nn.ReLU(inplace=True)
        # ])

        self.v_embedding = spectral_norm(nn.Linear(117, d_model))
        self.o_embedding = spectral_norm(nn.Linear(81, d_model))

        self.model = nn.Sequential(
            spectral_norm(nn.Linear(d_model * 3, d_model * 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(d_model * 2, d_model)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(d_model, 1)),
        )

    def forward(self, verb_logits, obj_logits, embeddings):
        """

        :param logits: verb logits for proposal.
        :param embeddings: the embedding of triplet before classifier.
        :return:
        """

        v_logit_emd = self.v_embedding(verb_logits)
        o_logit_emd = self.o_embedding(obj_logits)
        return self.model(torch.cat((v_logit_emd, o_logit_emd, embeddings), dim=-1)).squeeze_(-1)
