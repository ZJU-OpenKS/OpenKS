"""
@File    :    pooler_interp.py
@Time    :    2021/3/16 16:08
@Author  :    Bowen Cheng
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import os, sys
from pointnet2_modules import PointnetSAModuleVotes,PointnetSAMoudleAgg
import pointnet2_utils
# from pointnet2.pointnet2_utils import gather_operation
# from box_util import get_global_grid_points_of_rois
# from box_util import rotz_batch_pytorch
from pytorch_utils import SharedMLP


class ROIGridPooler(nn.Module):
    def __init__(
            self,
            grid_size,
            seed_feat_dim,
            rep_type,
            ray_density,
            revisit_method: str,
            interp_num: int,
            one_step_type: str,
    ):
        super(ROIGridPooler, self).__init__()

        self.grid_size = grid_size
        self.ray_density = ray_density
        self.rep_type = rep_type
        if self.rep_type == "grid":
            self.num_key_points = grid_size ** 3
        elif self.rep_type == "ray":
            self.num_key_points = ray_density * 6
        else:
            raise NotImplementedError

        self.revisit_method = revisit_method
        self.interp_num = interp_num  # Number of nearest neighbours for interpolation
        self.one_step_type = one_step_type  # Method used in one step visit
        if revisit_method == "set_abstraction":
            self.seed_aggregation = PointnetSAMoudleAgg(
                radius=0.2,
                nsample=16,
                mlp=[seed_feat_dim, 128, 64, 32],
                use_xyz=True,
                normalize_xyz=True
            )
            self.reduce_dim = torch.nn.Conv1d(self.num_key_points * 32, 128, 1)

        elif revisit_method == 'interpolation':
            self.mlp = SharedMLP(
                [seed_feat_dim + 3, 128, 128],
                bn=True
            )

        elif revisit_method == "one_step":
            if one_step_type in ['interp-coord', 'maxpool-coord']:
                self.mlp = SharedMLP(
                    [seed_feat_dim, 128, 128, 128],
                    bn=True
                )
            elif one_step_type == 'maxpool+coord':
                self.mlp = SharedMLP(
                    [seed_feat_dim + 3, 128, 128, 128],
                    bn=True
                )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def forward(self, end_points):
        # ---- recover Rep points ----
        # Recover heading angle
        batch_size, num_proposal, num_heading_bin = end_points['heading_scores'].shape

        pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # (B, num_proposal)
        pred_heading_residuals = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)).squeeze(-1)  # (B, num_proposal)

        if num_heading_bin != 1:  # for SUN RGBD dataset
            pred_heading = pred_heading_class.float() * (2 * np.pi / float(num_heading_bin)) + pred_heading_residuals  # (B, num_proposal)
            pred_heading = pred_heading % (2*np.pi)
        else:  # for ScanNetV2 dataset
            pred_heading = torch.zeros((batch_size, num_proposal)).cuda()
        end_points['heading_angle_0'] = pred_heading

        # Recover predicted distances
        pred_rois = end_points['rois_0']  # (B, num_proposal, 6)

        vote_xyz = end_points['aggregated_vote_xyz']  # (B, num_proposal, 3)
        # Recover back-projected points and Get key points
        if self.rep_type == 'ray':
            # (B, N, num_key_points, 3)
            # (B, N, 6, 3)
            key_points = ray_based_points(vote_xyz, pred_rois, pred_heading, self.ray_density)
        elif self.rep_type == 'grid':
            key_points = grid_based_points(vote_xyz, pred_rois, pred_heading, self.grid_size)
        else:
            raise NotImplementedError
        end_points['key_points'] = key_points  # (B, N, num_key_points, 3)
        # (B, N*num_key_points, 3)
        # key_points = key_points.view(batch_size, -1, 3)

        # ---- Pool seed features (Revisit) ----
        seed_xyz = end_points['seed_xyz']
        seed_features = end_points['seed_features']
        if self.revisit_method == "set_abstraction":
            features = self._SA_revisit(key_points, seed_xyz, seed_features)
        elif self.revisit_method == "interpolation":
            features = self._Interp_revisit(key_points, seed_xyz, seed_features, self.interp_num, vote_xyz)
        elif self.revisit_method == "one_step":
            features = self._One_step_revisit(key_points, seed_xyz, seed_features, self.interp_num, vote_xyz, type=self.one_step_type)
        else:
            raise NotImplementedError

        return features

    def _SA_revisit(self, key_points, seed_xyz, seed_features):
        assert self.revisit_method == "set_abstraction"
        batch_size, num_proposals, _, _ = key_points.shape

        # (bs, num_proposals*num_key_points, 3)
        key_points = key_points.view(batch_size, -1, 3)

        features = self.seed_aggregation(seed_xyz, key_points, seed_features)
        # (bs, mlp[-1], num_proposals, num_key_points)
        features = features.view(batch_size, -1, num_proposals, self.num_key_points)
        # (bs, mlp[-1], num_key_points, num_proposals)
        features = features.transpose(2, 3)
        # (bs, mlp[-1]*num_key_points, num_proposals)
        features = features.reshape(batch_size, -1, num_proposals)
        # (bs, 128, num_proposals)
        features = self.reduce_dim(features)

        return features

    def _Interp_revisit(self, rep_points, origin_xyz, origin_features, topK, center_xyz):
        """
        revisit seed points using interpolation
        Args:
            rep_points: (B, K, #rep, 3)
            origin_xyz: (B, 1024, 3)
            origin_features: (B, feat_size, 1024)
            topK:
            center_xyz: (B, K, 3)
        Returns:
        """
        assert self.revisit_method == "interpolation"

        # (B, 1024, feat_size)
        origin_features_transposed = origin_features.transpose(1,2).contiguous()
        feat_size = origin_features.shape[1]
        B, K, _, _ = rep_points.shape

        # (B, K * #rep, 3)
        rep_points  = rep_points.view(B, -1, 3)
        # (B, K * #rep, #seed)
        dist = nn_distance(rep_points, origin_xyz)
        # (B, K * #rep, topK)
        topK_dist, topK_idx = torch.topk(dist, topK, dim=-1, largest=False, sorted=True)

        # (B, K * #rep, 3)
        # offset: rep_point - center_xyz
        relative_points = rep_points - center_xyz.unsqueeze(2).repeat(1,1,self.num_key_points,1).contiguous().view(B,-1,3)

        weight = 1 / (topK_dist+1e-8)  # (B, K * #rep, topK)
        norm = torch.sum(weight, dim=2, keepdim=True)  # (B, K * #rep, 1)
        weight = weight / norm
        weight = weight.contiguous()  # (B, K * #rep, topK)

        # (B, K * #rep * topK, feat_size)
        interp_feats = torch.gather(
            origin_features_transposed,
            dim=1,
            index=topK_idx.view(B, -1, 1).repeat(1,1,feat_size).long()
        )
        # (B, K * #rep, topK, feat_size)
        interp_feats = interp_feats.view(B, -1, topK, feat_size)
        # (B, K * #rep, feat_size)
        interp_feats = torch.sum(interp_feats * weight.unsqueeze(-1), dim=2)
        # (B, feat_size, K * #rep)
        interp_feats = interp_feats.transpose(1,2)
        # (B, feat_size, K, #rep)
        interp_feats = interp_feats.view(B, feat_size, K, self.num_key_points)
        # (B, 3, K, #rep)
        relative_points = relative_points.transpose(1,2).view(B, -1, K, self.num_key_points)
        # (B, 3+feat_size, K, #rep)
        interp_feats = torch.cat([relative_points, interp_feats], dim=1)
        # (B, 128, K, #rep)
        interp_feats = self.mlp(interp_feats)
        # (B, 128, K)
        interp_feats = F.max_pool2d(interp_feats, kernel_size=[1, interp_feats.size(3)]).squeeze(-1)
        return interp_feats

    def _One_step_revisit(
            self,
            rep_points,
            origin_xyz,
            origin_features,
            topK: int,
            center_xyz,
            type: str,
    ):
        """
        Args:
            rep_points:
            origin_xyz:
            origin_features:
            topK:
            center_xyz:
            type: One step revisit type
                1. interp-coord (interpolation without coordinate)
                2. maxpool+coord (max pooling with coordinate)
                3. maxpool-coord (max pooling without coordinate)
        Returns:
        """
        assert self.revisit_method == "one_step"
        B, K, _, _ = rep_points.shape

        # (B, K * #rep, 3)
        rep_points = rep_points.view(B, -1, 3)
        # (B, K * #rep, #seed)
        dist = nn_distance(rep_points, origin_xyz)
        # (B, K * #rep, topK)
        topK_dist, topK_idx = torch.topk(dist, topK, dim=-1, largest=False, sorted=True)
        # (B, K, #rep, topK)
        topK_idx = topK_idx.view(B, K, -1, topK)
        # (B, K, #rep * topK)
        selected_idx = topK_idx.view(B, K, -1)

        # (B, K * #rep * topK, 3)
        selected_xyz = torch.gather(
            origin_xyz,
            dim=1,
            index=selected_idx.view(B, -1, 1).repeat(1,1,3).long()
        )
        # (B, K, #rep * topK, 3)
        selected_xyz = selected_xyz.view(B, K, -1, 3)
        # (B, K, #rep * topK, 3)
        relative_points = selected_xyz - center_xyz.unsqueeze(2)

        # (B, K * #rep * topK, feat_size)
        feat_size = origin_features.shape[1]
        selected_features = torch.gather(
            origin_features.transpose(1,2).contiguous(),
            dim=1,
            index=selected_idx.view(B, -1, 1).repeat(1,1,feat_size).long()
        )
        # (B, K, #rep * topK, feat_size)
        selected_features = selected_features.view(B, K, -1, feat_size)

        if type == 'interp-coord':
            # (B, K, #rep * topK)
            relative_dist = torch.sqrt(torch.sum(relative_points ** 2, dim=-1))
            # (B, K, #rep * topK)
            weight = 1 / (relative_dist + 1e-8)
            # (B, K, 1)
            norm = torch.sum(weight, dim=2, keepdim=True)
            # (B, K, #rep * topK)
            weight = weight / norm
            weight = weight.contiguous()
            # (B, K, feat_size)
            interp_features = torch.sum(selected_features*weight.unsqueeze(-1), dim=2)
            # (B, feat_size, K, 1)
            interp_features = interp_features.transpose(1,2).unsqueeze(-1)
            # (B, 128, K, 1)
            interp_features = self.mlp(interp_features)
            # (B, 128, K)
            ret_features = interp_features.squeeze(-1)

        elif type in ['maxpool-coord', 'maxpool+coord']:
            # (B, feat_size, K, #rep * topK)
            selected_features = selected_features.permute(0, 3, 1, 2)
            # (B, 3, K, #rep * topK)
            relative_points = relative_points.permute(0, 3, 1, 2)
            if type == 'maxpool+coord':
                # (B, 3+feat_size, K, #rep * topK)
                selected_features = torch.cat([relative_points, selected_features], dim=1)
            # (B, 128, K, #rep * topK)
            selected_features = self.mlp(selected_features)
            # (B, 128, K)
            ret_features = F.max_pool2d(selected_features, kernel_size=[1, selected_features.size(3)]).squeeze(-1)
        else:
            raise NotImplementedError

        return ret_features



def nn_distance(pc1, pc2):
    """
    Compute distance
    Args:
        pc1: (B, N, C)
        pc2: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)  # (B, N, M, C)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)  # (B, N, M, C)
    pc_diff = pc1_expand_tile - pc2_expand_tile  # (B, N, M, C)

    pc_dist = torch.sqrt(torch.sum(pc_diff**2, dim=-1))  # (B, N, M)
    return pc_dist


def ray_based_points(center, rois, angle, density):
    batch_size, num_proposal, _ = rois.shape  # (B, N, 6)
    R = rotz_batch_pytorch(angle).reshape(-1, 3, 3)  # Rotation matrix ~ (B*N, 3, 3)
    # Convert param pairs (rois, angle) to point locations
    num_key_points = density * 6
    back_proj_points = torch.zeros((batch_size, num_proposal, 6, 3)).cuda()  # (B, N, 6, 3)
    back_proj_points[:, :, 0, 0] = - rois[:, :, 0]  # back
    back_proj_points[:, :, 1, 1] = - rois[:, :, 1]  # left
    back_proj_points[:, :, 2, 2] = - rois[:, :, 2]  # down
    back_proj_points[:, :, 3, 0] = rois[:, :, 3]  # front
    back_proj_points[:, :, 4, 1] = rois[:, :, 4]  # right
    back_proj_points[:, :, 5 ,2] = rois[:, :, 5]  # up
    back_proj_points = back_proj_points.reshape(-1, 6, 3)  # (B*N, 6, 3)
    local_points = [back_proj_points*float(i/density) for i in range(density, 0, -1)]
    local_points = torch.stack(local_points, dim=1)  # (B*N, density, 6, 3)
    local_points = local_points.transpose(1,2).contiguous()  # (B*N, 6, density, 3)
    local_points = local_points.reshape(-1, num_key_points, 3)  # (B*N, num_key_points, 3)
    local_points_rotated = torch.matmul(local_points, R)  # (B*N, num_key_points, 3)
    center = center.reshape(batch_size*num_proposal, 1, 3)  # (B*N, 1, 3)
    global_points = local_points_rotated + center  # (B*N, num_key_points, 3)
    global_points = global_points.reshape(batch_size, num_proposal, num_key_points, 3)  # (B, N, num_key_points, 3)

    return global_points

def grid_based_points(center, rois, heading_angle, grid_size):
    B = heading_angle.shape[0]
    N = heading_angle.shape[1]
    # Rotation matrix ~ (B*N, 3, 3)
    R = rotz_batch_pytorch(heading_angle.float()).view(-1, 3, 3)
    # (B*N, gs**3, 3)
    local_grid_points = get_dense_grid_points(rois, B*N, grid_size)
    # (B*N, gs**3, 3) ~ add Rotation
    local_grid_points = torch.matmul(local_grid_points, R)
    # (B*N, gs**3, 3)
    global_roi_grid_points = local_grid_points + center.view(B*N, 3).unsqueeze(1)
    # (B, N, gs**3, 3)
    global_roi_grid_points = global_roi_grid_points.view(B, N, -1, 3)
    return global_roi_grid_points


def rotz_batch_pytorch(t):
    """
    Rotation about z-axis
    :param t:
    :return:
    """
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3])).cuda()
    c = torch.cos(t)
    s = torch.sin(t)
    # Transposed rotation matrix for x'A' = (Ax)'
    # [[cos(t), -sin(t), 0],
    #  [sin(t), cos(t),  0],
    #  [0,      0,       1]]
    output[..., 0, 0] = c
    output[..., 0, 1] = -s
    output[..., 1, 0] = s
    output[..., 1, 1] = c
    output[..., 2, 2] = 1
    return output


def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
    """
    :param rois: (B, num_proposal, 6) ~ back/left/down/front/right/up
    :param batch_size_rcnn: B*num_proposal
    :param grid_size:
    :return:
    """
    faked_features = rois.new_ones((grid_size, grid_size, grid_size))  # alis gs for grid_size
    dense_idx = faked_features.nonzero()  # (gs**3, 3) [x_idx, y_idx, z_idx]
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (batch_size_rcnn, gs**3, 3)

    rois_center = rois[:, :, 0:3].view(-1, 3)  # (batch_size_rcnn, 3)
    local_rois_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (B, num_proposal, 3)
    local_rois_size = local_rois_size.view(-1, 3)  # (batch_size_rcnn, 3)
    roi_grid_points = (dense_idx + 0.5) / grid_size * local_rois_size.unsqueeze(dim=1)  # (batch_size_rcnn, gs**3, 3)
    roi_grid_points = roi_grid_points - rois_center.unsqueeze(dim=1)  # (batch_size_rcnn, gs**3, 3)
    return roi_grid_points
