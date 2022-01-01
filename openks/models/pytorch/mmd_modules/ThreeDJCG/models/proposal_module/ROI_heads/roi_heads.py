"""
@File    :    roi_heads.py
@Time    :    2021/3/16 16:08
@Author  :    Bowen Cheng
"""
# BRNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys


class StandardROIHeads(nn.Module):
    def __init__(
            self,
            num_heading_bin,
            num_class,
            use_exp=True,
            use_centerness=False,
            obj_loss=True,
            back_trace_type=None,
            seed_feat_dim=256,
            density=1,
            rep_type='ray',
            revisit_method='one_step',
            interp_num=16,
            one_step_type='maxpool-coord',
    ):
        super(StandardROIHeads, self).__init__()

        self.num_heading_bin = num_heading_bin
        self.num_class = num_class
        # TODO NOT USED
        self.use_exp = use_exp
        self.use_centerness = use_centerness
        self.obj_loss = obj_loss
        self.seed_feat_dim=seed_feat_dim
        self.back_trace_type=back_trace_type

        convs = [
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        ]
        self.convs = nn.Sequential(*convs)

        self.objectness_predictor = nn.Conv1d(128, 2, kernel_size=1)
        self.box_predictor = nn.Conv1d(128, 6, kernel_size=1)
        if self.num_class:
            self.sem_cls_predictor = nn.Conv1d(128, num_class, kernel_size=1)
        self.heading_cls_predictor = nn.Conv1d(128, num_heading_bin, kernel_size=1)
        self.heading_reg_predictor = nn.Conv1d(128, num_heading_bin, kernel_size=1)

        predictors = [self.objectness_predictor, self.box_predictor]

        for layer in convs:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for predictor in predictors:
            nn.init.normal_(predictor.weight, std=0.001)
            if predictor.bias is not None:
                nn.init.constant_(predictor.bias, 0)

        # revisit
        if self.back_trace_type is not None:
            raise NotImplementedError()
            from .pooler_interp import ROIGridPooler
            self.density = density
            self.rep_type = rep_type
            if self.rep_type == 'ray':
                self.num_key_points = 6*density
            elif self.rep_type == 'grid':
                self.num_key_points = density**3
            else:
                raise NotImplementedError
            print('Representative points configuration: %s, density: %d' % (self.rep_type, self.density))
            self.pooler = ROIGridPooler(
                density,
                seed_feat_dim,
                rep_type,
                density,
                revisit_method,
                interp_num,
                one_step_type
            )

    def forward(
            self,
            ROI_features,
            data_dict
    ):
        batch_size, num_proposal, _ = ROI_features.shape

        # ---------- box_proposal_0 ----------------
        x = self.convs(ROI_features)

        # (batch_size, num_proposal, 2)
        pred_objectness_logits = self.objectness_predictor(x).permute(0,2,1)

        # (batch_size, num_proposal, 6)
        pred_box_reg = self.box_predictor(x).permute(0,2,1)
        pred_box_reg = pred_box_reg.exp()  # add exp transform

        # (batch_size, num_proposal, num_heading_bin)
        pred_heading_cls = self.heading_cls_predictor(x).permute(0,2,1)
        pred_heading_reg = self.heading_reg_predictor(x).permute(0,2,1)

        # class
        if self.num_class:
            pred_sem_cls = self.sem_cls_predictor(x).permute(0,2,1)
            data_dict['sem_cls_scores'] = pred_sem_cls

        # store to data_dict dict
        data_dict['heading_scores'] = pred_heading_cls  # (B, N, num_heading_bin)
        data_dict['heading_residuals_normalized'] = pred_heading_reg  # (B, N, num_heading_bin)
        data_dict['heading_residuals'] = pred_heading_reg * (np.pi / self.num_heading_bin)
        data_dict['rois'] = pred_box_reg  # (B, N, 6)
        data_dict['objectness_scores'] = pred_objectness_logits  # (B, N, 2)

        # # ROIGrid pooling
        # if self.back_trace_type is not None:
        #     new_seed_features = self.pooler(data_dict)  # (B, 128, num_proposal)
        # else:
        #     new_seed_features = ROI_features
        # data_dict['new_seed_features'] = new_seed_features

        return data_dict

