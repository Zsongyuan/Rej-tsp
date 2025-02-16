# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# BRIEF Object candidate point prediction from seed point features.
class PointsObjClsModule(nn.Module):

    def __init__(self, seed_feature_dim):
        """
        Object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn2 = nn.BatchNorm1d(self.in_dim)
        self.conv3 = nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class GeneralSamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_operation(
            xyz_flipped, sample_inds
        ).transpose(1, 2).contiguous()
        new_features = gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds

# BRIEF predict object position, size and class.
class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)

# BRIEF predict head.
class ClsAgnosticPredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal,
                 seed_feat_dim=256, objectness=True, heading=True,
                 compute_sem_scores=True):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim
        self.objectness = objectness
        self.heading = heading
        self.compute_sem_scores = compute_sem_scores

        if objectness:
            self.objectness_scores_head = ThreeLayerMLP(seed_feat_dim, 1)
        self.center_residual_head = ThreeLayerMLP(seed_feat_dim, 3)
        if heading:
            self.heading_class_head = nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_pred_head = ThreeLayerMLP(seed_feat_dim, 3)
        if compute_sem_scores:
            self.sem_cls_scores_head = ThreeLayerMLP(seed_feat_dim, self.num_class)

    def forward(self, bbox_preds, cls_preds, mask, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = features  # ([B, C=288, num_proposal=256])

        if self.objectness:
            objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
            end_points[f'{prefix}objectness_scores'] = objectness_scores.squeeze(-1)

        # step 1. center
        center_residual = self.center_residual_head(net).transpose(2, 1)    # ([B, 256, 3])
        center = base_xyz + center_residual                                 # (B, num_proposal, 3)
        if self.heading:
            heading_scores = self.heading_class_head(net).transpose(2, 1)
            end_points[f'{prefix}heading_scores'] = heading_scores

        # step 2. size
        pred_size = self.size_pred_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # step 2. class
        if self.compute_sem_scores:
            sem_cls_scores = self.sem_cls_scores_head(features).transpose(2, 1)  # (B, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}pred_size'] = pred_size
        
        bbox = torch.cat([center,pred_size],dim=-1)
        bbox_pred, cls_pred = [],[]
        for b_i in range(len(bbox)):
            bbox_pred.append(bbox[b_i][~mask[b_i]])
            cls_pred.append(heading_scores[b_i][~mask[b_i]])
        bbox_preds.append(bbox_pred)
        cls_preds.append(cls_pred)

        if self.compute_sem_scores:
            end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores
        return center, pred_size
    
def optional_repeat(value, times):
    """ helper function, to repeat a parameter's value many times
    :param value: an single basic python type (int, float, boolean, string), or a list with length equals to times
    :param times: int, how many times to repeat
    :return: a list with length equal to times
    """
    if type(value) is not list:
        value = [value]

    if len(value) != 1 and len(value) != times:
        raise ValueError('The value should be a singleton, or be a list with times length.')

    if len(value) == times:
        return value # do nothing

    return np.array(value).repeat(times).tolist()

class MLP(nn.Module):
    """ Multi-near perceptron. That is a k-layer deep network where each layer is a fully-connected layer, with
    (optionally) batch-norm, a non-linearity and dropout. The last layer (output) is always a 'pure' linear function.
    """
    def __init__(self, in_feat_dims, out_channels, b_norm=True, dropout_rate=0,
                 non_linearity=nn.ReLU(inplace=True), closure=None):
        """Constructor
        :param in_feat_dims: input feature dimensions
        :param out_channels: list of ints describing each the number hidden/final neurons. The
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        :param closure: optional nn.Module to use at the end of the MLP
        """
        super(MLP, self).__init__()

        n_layers = len(out_channels)
        dropout_rate = optional_repeat(dropout_rate, n_layers-1)
        b_norm = optional_repeat(b_norm, n_layers-1)

        previous_feat_dim = in_feat_dims
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:
                if b_norm[depth]:
                    all_ops.append(nn.BatchNorm1d(out_dim))

                if non_linearity is not None:
                    all_ops.append(non_linearity)

                if dropout_rate[depth] > 0:
                    all_ops.append(nn.Dropout(p=dropout_rate[depth]))

            previous_feat_dim = out_dim

        if closure is not None:
            all_ops.append(closure)

        self.net = nn.Sequential(*all_ops)

    def __call__(self, x):
        return self.net(x)
