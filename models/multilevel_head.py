import numpy as np

import MinkowskiEngine as ME

import torch,time
from mmcv.ops import nms3d, nms3d_normal
from torch import nn
import torch.nn.functional as F

from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from .axis_aligned_iou_loss import AxisAlignedIoULoss2
from mmdet.models.losses import FocalLoss
from .trans_modules import (BiEncoder, BiEncoderLayer, PositionEmbeddingLearned)

import logging

class MinkowskiFeatureFusionBlock(nn.Module):
    """
    Block to fuse backbone features with text features in Minkowski space.
    """
    def __init__(self, backbone_channels, text_channels, output_channels, dimension=3):
        super(MinkowskiFeatureFusionBlock, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            backbone_channels + text_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.norm = ME.MinkowskiBatchNorm(output_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, backbone_feats, text_feats):
        # Extract batch indices from the coordinates of backbone features
        batch_indices = backbone_feats.C[:, 0].long()  # Last column is batch index
        
        # Repeat text features for each point in the corresponding batch
        repeated_text_feats = text_feats[batch_indices]  # Use indexing to repeat text features
        
        # Combine the backbone and text features
        combined_features = torch.cat([backbone_feats.F, repeated_text_feats], dim=1)
        combined_feats = ME.SparseTensor(
            features=combined_features,
            coordinate_map_key=backbone_feats.coordinate_map_key,
            coordinate_manager=backbone_feats.coordinate_manager
        )
        
        # Convolution and normalization
        x = self.conv(combined_feats)
        x = self.norm(x)
        return self.relu(x)
    
def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class TSPHead(nn.Module):
    def __init__(self,
                 n_classes=1,
                 in_channels=(128, 128, 128),
                 out_channels=128,
                 n_reg_outs=6,
                 voxel_size=.01,
                 pts_prune_threshold=(1200,4000),
                 top_pts_threshold=32,
                 volume_threshold=27,
                 r=(13,13),
                 assign_type='volume',
                 prune_threshold=(0.3,0.7),
                 com_threshold = 0.15,
                 train_cfg=None,
                 test_cfg=dict(nms_pre=1, iou_thr=.5, score_thr=.01, ans_thr=0.5, ans_temp=1.0),
                 keep_loss_weight = 1.0,
                 bbox_loss_weight = 1.0,
                 lambda_a=0.25,
                 lambda_bg=0.1):
        super(TSPHead, self).__init__()
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.assign_type = assign_type
        self.volume_threshold = volume_threshold
        self.r = r
        self.prune_threshold = prune_threshold
        self.keep_loss_weight = keep_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.lambda_a = lambda_a
        self.lambda_bg = lambda_bg
        self.assigner = TR3DAssigner(top_pts_threshold=32, label2level=[0])
        self.bbox_loss = AxisAlignedIoULoss2(mode='diou', reduction='none')
        self.cls_loss = FocalLoss(reduction='none', use_sigmoid=True)
        self.com_loss = FocalLoss(reduction='none', use_sigmoid=True)
        self.keep_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_samples = (3200,320)
        self.num_samples_com = 2400
        self.com_threshold = com_threshold
        self.random_prune_threshold = (1200,4000)
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)
        self.ans_fc = nn.Linear(out_channels, 1)


    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)
        ])
        self.pos_embed = PositionEmbeddingLearned(3, 128)
        bi_layer0 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer1 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer2 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        self.keep_trans = nn.ModuleList([BiEncoder(bi_layer0, 2), BiEncoder(bi_layer1, 2)])
        self.com_trans = BiEncoder(bi_layer2, 2)
        self.pruning = ME.MinkowskiPruning()
        self.com_cls = nn.Conv1d(128, 1, kernel_size=1, bias=True)


        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            self.__setattr__(
                        f'lateral_block_{i}',
                        self.make_block(in_channels[i], in_channels[i]))
            if i == 0:
                self.__setattr__(
                    f'out_block_{i}',
                    self.make_block(in_channels[i], out_channels))

        self.fuse = MinkowskiFeatureFusionBlock(128, 128, 128)


    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
        nn.init.normal_(self.ans_fc.weight, std=.01)
        nn.init.constant_(self.ans_fc.bias, bias_init_with_prob(.4))

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)

        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) \
                and ('keep_conv' not in n) and ('loss' not in n) and ('ans_fc' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)       
    

    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores = self.cls_conv(x)
        cls_pred = scores.features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:]* self.voxel_size)
        return bbox_preds, cls_preds, points

    def _compute_answerability(self, out, text_feats):
        """Compute answerability logits from proposal features."""
        ans_scores = []
        for b, perm in enumerate(out.decomposition_permutations):
            feats = out.features[perm]
            if feats.numel() == 0:
                pooled = feats.new_zeros(feats.shape[1])
            else:
                feats = F.normalize(feats, p=2, dim=1)
                txt = F.normalize(text_feats[b, 0], p=2, dim=0)
                weights = torch.matmul(feats, txt)
                weights = torch.softmax(weights, dim=0)
                pooled = (feats * weights.unsqueeze(1)).sum(0)
            ans_scores.append(self.ans_fc(pooled))
        return torch.stack(ans_scores).squeeze(-1)

    def forward(self, x,text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc=None):
        bboxes_level = []
        bboxes_state = []
        if self.assign_type == 'volume':
            for idx in range(len(img_metas)):
 
                bbox_all = gt_all_bbox_new[idx]
                bbox_level = torch.ones([bbox_all.shape[0], 1])
                bbox_state_all = torch.cat((bbox_level, bbox_all.gravity_center, bbox_all.tensor[:, 3:]), dim=1)

                bbox_gt = gt_bboxes[idx]
                bbox_state_gt = torch.cat((bbox_gt.gravity_center, bbox_gt.tensor[:, 3:]), dim=1)                
                bbox_auxi = auxi_bbox[idx]
                bbox_state_auxi = torch.cat((bbox_auxi.gravity_center, bbox_auxi.tensor[:, 3:]), dim=1)
                bbox_state_auxi_gt = torch.cat((bbox_state_gt, bbox_state_auxi), dim=0)
                bbox_level = torch.zeros([bbox_state_auxi_gt.shape[0], 1])
                bbox_state_auxi_gt = torch.cat((bbox_level, bbox_state_auxi_gt), dim=1)
                
                bbox_state = torch.cat((bbox_state_all, bbox_state_auxi_gt), dim=0)
                
                bboxes_level.append(bbox_state[:,[0]])
                bboxes_state.append(bbox_state)
        
        bbox_preds, cls_preds, points = [], [], []
        keep_gts, keep_preds = [], []
        inputs = x[1:]
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):  # 2,1,0
            if i == 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                          coordinate_manager=x.coordinate_manager)
                x = x + x_level
            elif i == 0:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                prune_threshold_ = np.random.randint(self.random_prune_threshold[0], self.random_prune_threshold[1])
                self.pts_prune_threshold = (prune_threshold_, self.pts_prune_threshold[1])
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                          coordinate_manager=x.coordinate_manager)
                x_ori = x + x_level

                sampled_coords, sampled_features, original_indices = [], [], []

                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]),
                                                                          dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0)
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                             dtype=inputs[0].coordinates.dtype).to(inputs[0].device)],
                            dim=0)
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:, :, 1:] * self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :, 0] != -1
                com_pred_training = [com_pred[k][valid_mask[k]] for k in range(len(com_pred))]
                com_coords_training = [sampled_coords[k][valid_mask[k]][:, 1:] * self.voxel_size for k in range(len(com_pred))]
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > self.com_threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]

                x_com_features = x.features_at_coordinates(sampled_coords.float())
                x_com_features = x_com_features + sampled_features
                x = ME.SparseTensor(features=torch.cat((x_ori.features, x_com_features), dim=0),
                                    coordinates=torch.cat((x_ori.coordinates, sampled_coords), dim=0),
                                    coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
            if i > 0: # 2,1
                sampled_coords,sampled_features, original_indices = [],[],[]
                prune_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(x.device)
                for permutation in x.decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples[i-1]:
                        choice = torch.randperm(len(permutation))[:self.num_samples[i-1]]
                        choice = torch.sort(choice).values
                        sampled_features.append(x.features[permutation][choice])
                        sampled_coords.append(x.coordinates[permutation][choice])
                        prune_mask[permutation[choice]] = True
                    else:
                        padding_size = self.num_samples[i-1] - len(permutation)      
                        padded_features = torch.cat(
                            [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                                                  dtype=x.features.dtype).to(x.device)], dim=0) 
                        padded_coords = torch.cat(
                            [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                     dtype=x.coordinates.dtype).to(x.device)], 
                                                                     dim=0)  
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                        prune_mask[permutation] = True
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.keep_trans[i-1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:, :, 1:] * self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                valid_mask = sampled_coords[:, :, 0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]

                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords,
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)

                prune_mask = self._get_keep_voxel(x, i + 1, bboxes_state, img_metas)
                keep_gt = []
                for permutation in x.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)

                keep_scores = self.keep_conv[i-1](x)  # 1 MLP
                prune_training_keep = ME.SparseTensor(
                    -keep_scores.features,
                    coordinate_map_key=keep_scores.coordinate_map_key,
                    coordinate_manager=keep_scores.coordinate_manager)

                keep_pred = keep_scores.features
                keeps = []

                try:
                    for permutation in x.decomposition_permutations:
                        keeps.append(keep_pred[permutation])
                except Exception as e:
                    raise RuntimeError("failed to collect keep predictions") from e
                keep_preds.append(keeps)

                # Prune current features using freshly computed scores
                # ``pts_prune_threshold`` is defined for two pruning stages,
                # corresponding to ``i=2`` and ``i=1`` in the loop above.
                # The previous implementation used the loop index ``i``
                # directly, which caused an ``IndexError`` when ``i`` was 2
                # because ``pts_prune_threshold`` only has two elements
                # (indices 0 and 1). We offset the index by 1 so that the
                # thresholds map correctly to the pruning stages:
                #   i=2 -> threshold[1]
                #   i=1 -> threshold[0]
                x = self._prune_training(x, prune_training_keep, i - 1)

            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
        out = self.fuse(out, text_feats[:, 0])
        ans_pred = self._compute_answerability(out, text_feats)
        bbox_pred, cls_pred, point = self._forward_single(out)
        return [bbox_pred], [cls_pred], [point], keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training, ans_pred
    

    def _prune_inference(self, x, scores, layer_id):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            prune_mask = scores.new_zeros(
                (len(scores)), dtype=torch.bool)

            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > self.prune_threshold[layer_id]
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True                 
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        return x


    def _prune_training(self, x, scores, layer_id):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold[layer_id])
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x


    @torch.no_grad()
    def _get_keep_voxel(self, input, cur_level, bboxes_state, input_metas):
        bboxes = []
        for size in range(len(input_metas)):
            bboxes.append([])
        for idx in range(len(input_metas)):
            for n in range(len(bboxes_state[idx])):
                if bboxes_state[idx][n][0] < (cur_level - 1):    
                    bboxes[idx].append(bboxes_state[idx][n])
        idx = 0
        mask = []
        l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
        for idx, permutation in enumerate(input.decomposition_permutations):
            point = input.coordinates[permutation][:, 1:]* self.voxel_size
            if len(bboxes[idx]) != 0:
                point = input.coordinates[permutation][:, 1:]* self.voxel_size
                boxes = bboxes[idx]
                level = 3
                bboxes_level = [[] for _ in range(level)]
                for n in range(len(boxes)):
                    for l in range(level):
                        if boxes[n][0] == l:
                            bboxes_level[l].append(boxes[n])
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                for l in range(level):
                    if len(bboxes_level[l]) != 0:
                        point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                        boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                        boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                        shift = torch.stack(
                            (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                            point_l[..., 2] - boxes_l[..., 3]),
                            dim=-1).permute(1, 0, 2)
                        shift = rotation_3d_in_axis(
                            shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                        centers = boxes_l[..., 1:4] + shift
                        up_level_l = self.r[cur_level-2] 
                        dx_min = centers[..., 0] - boxes_l[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2 
                        dy_min = centers[..., 1] - boxes_l[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dz_min = centers[..., 2] - boxes_l[..., 3] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2


                        distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                        inside_box_condition = distance.min(dim=-1).values > 0
                        inside_box_condition = inside_box_condition.sum(dim=1)
                        inside_box_condition = inside_box_condition >= 1
                        inside_box_conditions += inside_box_condition
                mask.append(inside_box_conditions)
            else:
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                mask.append(inside_box_conditions)

        prune_mask = torch.cat(mask)
        prune_mask = prune_mask.to(input.device)
        return prune_mask
    

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)


    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta,
                     com_pred,com_coords):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        cls_targets = cls_preds.new_zeros((len(cls_preds), n_classes))
        if pos_mask.any():
            cls_targets[pos_mask, gt_labels[assigned_ids[pos_mask]]] = 1

        cls_loss = self.cls_loss(cls_preds, cls_targets)
        
        assigned_ids_com = self.assigner.assign([com_coords], gt_bboxes, gt_labels, img_meta)
        # com loss
        pos_mask_com = assigned_ids_com >= 0
        com_targets = com_pred.new_zeros((len(com_pred), n_classes))
        if pos_mask_com.any():
            com_targets[pos_mask_com, gt_labels[assigned_ids_com[pos_mask_com]]] = 1
        com_loss = self.com_loss(com_pred, com_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))            
        else:
            bbox_loss = None
        return bbox_loss, cls_loss, pos_mask, com_loss, pos_mask_com


    def _loss(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas,
            keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training, ans_preds):
        """
        改进的loss计算函数
        """
        bbox_losses, cls_losses, pos_masks, com_losses, pos_masks_com = [], [], [], [], []
        ans_losses, bg_losses = [], []

        # Keep loss calculation
        keep_losses = 0
        for i in range(len(img_metas)):
            k_loss = 0
            keep_pred = [x[i] for x in keep_preds]
            keep_gt = [x[i] for x in keep_gts]
            for j in range(len(keep_pred)):
                pred = keep_pred[j]
                gt = keep_gt[j].float().unsqueeze(1)

                if pred.size(0) != gt.size(0):
                    raise RuntimeError("keep pred and gt size mismatch")

                if gt.sum() != 0:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=gt.sum())
                    k_loss = torch.mean(keep_loss) / 3 + k_loss
                else:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=gt.shape[0])
                    k_loss = torch.mean(keep_loss) / 3 + k_loss

            keep_losses = keep_losses + k_loss

        # Per-image loss calculation
        for i in range(len(img_metas)):
            # answerability loss
            ans_target = ans_preds.new_tensor(0.0 if img_metas[i].get('is_negative', False) else 1.0)
            ans_losses.append(F.binary_cross_entropy_with_logits(ans_preds[i], ans_target))

            if img_metas[i].get('is_negative', False):
                # hard negative mining for G-head
                preds = torch.cat([p[i] for p in cls_preds])
                if preds.numel() > 0:
                    obj_scores, obj_labels = preds.sigmoid().max(dim=1)
                    k = min(max(1, int(0.01 * obj_scores.numel())), 50)
                    hard_ids = obj_scores.topk(k).indices
                    hard_logits = preds[hard_ids, obj_labels[hard_ids]]
                    bg_losses.append(F.binary_cross_entropy_with_logits(
                        hard_logits,
                        hard_logits.new_zeros(hard_logits.shape),
                        reduction='mean'))
                continue

            bbox_loss, cls_loss, pos_mask, com_loss, pos_mask_com = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                com_pred=com_pred_training[i],
                com_coords=com_coords_training[i])

            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            com_losses.append(com_loss)
            pos_masks.append(pos_mask)
            pos_masks_com.append(pos_mask_com)

        # 计算最终loss，确保不会出现nan
        if bbox_losses:
            final_bbox_loss = self.bbox_loss_weight * torch.mean(torch.cat(bbox_losses))
        else:
            final_bbox_loss = torch.tensor(0.0, device=cls_preds[0][0].device, requires_grad=True)

        device = cls_preds[0][0].device

        if pos_masks:
            total_pos = torch.stack([m.float().sum() for m in pos_masks]).sum()
        else:
            total_pos = torch.tensor(0.0, device=device)

        if cls_losses and total_pos > 0:
            total_cls = torch.stack([l.sum() for l in cls_losses]).sum()
            final_cls_loss = total_cls / total_pos
        else:
            final_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if pos_masks_com:
            total_pos_com = torch.stack([m.float().sum() for m in pos_masks_com]).sum()
        else:
            total_pos_com = torch.tensor(0.0, device=device)

        if com_losses and total_pos_com > 0:
            total_com = torch.stack([l.sum() for l in com_losses]).sum()
            final_com_loss = total_com / total_pos_com
        else:
            final_com_loss = torch.tensor(0.0, device=device, requires_grad=True)

        final_ans_loss = self.lambda_a * torch.mean(torch.stack(ans_losses))
        if bg_losses:
            final_bg_loss = self.lambda_bg * torch.mean(torch.stack(bg_losses))
        else:
            final_bg_loss = final_ans_loss.new_tensor(0.0)

        # 返回所有loss组件
        loss_dict = {
            'bbox_loss': final_bbox_loss,
            'cls_loss': final_cls_loss,
            'keep_loss': self.keep_loss_weight * keep_losses / len(img_metas),
            'com_loss': final_com_loss,
            'ans_loss': final_ans_loss,
            'bg_loss': final_bg_loss,
        }
        
        # 添加统计信息（可选）
        loss_dict['num_pos'] = total_pos.item() if hasattr(total_pos, 'item') else total_pos
        loss_dict['num_pos_com'] = total_pos_com.item() if hasattr(total_pos_com, 'item') else total_pos_com
        
        return loss_dict

    def forward_train(self, x, text_feats, text_attention_mask, gt_bboxes, 
                     gt_labels, gt_all_bbox_new, auxi_bbox, img_metas, pc=None):
        """Forward training with rejection loss support"""
        # 执行forward获取预测
        bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level, \
        com_pred_training, com_coords_training, ans_preds = self(
            x, text_feats, text_attention_mask, gt_bboxes, gt_labels,
            gt_all_bbox_new, auxi_bbox, img_metas, pc
        )

        losses = self._loss(
            bbox_preds, cls_preds, points,
            gt_bboxes, gt_labels, img_metas,
            keep_preds, keep_gts, bboxes_level,
            com_pred_training, com_coords_training, ans_preds
        )

        return losses


    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg['score_thr']
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg['iou_thr'])
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg['nms_pre'] > 0:
            _, ids = max_scores.topk(self.test_cfg['nms_pre'])
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        labels = boxes.new_zeros((1, ),dtype=int)
        boxes = img_meta['box_type_3d'](boxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        return boxes, scores, labels


    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results


    def forward_test(self, x, text_feats, text_attention_mask, img_metas, pc=None, gt_bboxes=None):
        inputs = x[1:]
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        prune_inference = None
        
        for i in range(len(inputs) - 1, -1, -1):
            if i ==1:
                if prune_inference is not None:
                    x = self._prune_inference(x, prune_inference, i)
                    if x is None:
                        raise RuntimeError("no points left after pruning at layer 1")
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)
                x_level = ME.SparseTensor(
                    features=x_level_features,
                    coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager,
                )
                x = x + x_level
            elif i ==0:
                if prune_inference is not None:
                    x = self._prune_inference(x, prune_inference, i)
                    if x is None:
                        raise RuntimeError("no points left after pruning at layer 0")
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)
                x_level = ME.SparseTensor(
                    features=x_level_features,
                    coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager,
                )
                x_ori = x + x_level
        
                sampled_coords,sampled_features, original_indices = [],[],[]
                
                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)      
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]), 
                                                                  dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0) 
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                     dtype=inputs[0].coordinates.dtype).to(inputs[0].device)], 
                                                                     dim=0)  
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > self.com_threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]                
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]                   
                
                x_com_features = x.features_at_coordinates(sampled_coords.float())     
                x_com_features = x_com_features + sampled_features           
                x = ME.SparseTensor(features=torch.cat((x_ori.features,x_com_features),dim=0), 
                                    coordinates=torch.cat((x_ori.coordinates,sampled_coords),dim=0), 
                                    coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
                
            if i > 0:
                sampled_coords,sampled_features = [],[]
                len_x = []
                for permutation in x.decomposition_permutations:
                    len_x.append(len(x.coordinates[permutation]))
                max_len_x = int(torch.tensor(len_x).max())
                if len(len_x)>1:
                    for permutation in x.decomposition_permutations:
                        if len(permutation) > max_len_x:
                            choice = torch.randperm(len(permutation))[:max_len_x]
                            choice = torch.sort(choice).values
                            sampled_features.append(x.features[permutation][choice])
                            sampled_coords.append(x.coordinates[permutation][choice])
                        else:
                            padding_size = max_len_x - len(permutation)      
                            padded_features = torch.cat(
                                [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                                                    dtype=x.features.dtype).to(x.device)], dim=0) 
                            padded_coords = torch.cat(
                                [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                        dtype=x.coordinates.dtype).to(x.device)], 
                                                                        dim=0)   
                            sampled_features.append(padded_features)
                            sampled_coords.append(padded_coords)
                else:
                    for permutation in x.decomposition_permutations:
                        sampled_features.append(x.features[permutation])
                        sampled_coords.append(x.coordinates[permutation])                        
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.keep_trans[i-1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords, 
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)
                keep_scores = self.keep_conv[i-1](x)
                prune_inference = keep_scores.features

            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
        start_time = time.time()
        out = self.fuse(out, text_feats[:, 0])
        ans_pred = self._compute_answerability(out, text_feats)
        bbox_pred, cls_pred, point = self._forward_single(out)
        results = self._get_bboxes([bbox_pred], [cls_pred], [point], img_metas)
        temp = self.test_cfg.get('ans_temp', 1.0)
        p_ans = torch.sigmoid(ans_pred / temp)
        gated_results = []
        for b, (bbox, score, label) in enumerate(results):
            if p_ans[b] < self.test_cfg.get('ans_thr', 0.5):
                empty_bbox = bbox.new_zeros((0, bbox.shape[1]))
                empty_score = score.new_zeros((0,))
                empty_label = label.new_zeros((0,), dtype=torch.long)
                gated_results.append((empty_bbox, empty_score, empty_label))
            else:
                gated_results.append((bbox, score, label))
        head_time = time.time() - start_time
        return gated_results, head_time

class TR3DAssigner:
    def __init__(self, top_pts_threshold, label2level):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: only closest object to point
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds