import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
import MinkowskiEngine as ME
from .mink_resnet import TSPBackbone
from .tr3d_neck import TR3DNeck
from .multilevel_head import TSPHead
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes
from mmdet3d.structures import bbox3d2result
import time
import pdb
    
class BeaUTyDETR(nn.Module):
    """
    3D language grounder.
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=128, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True, voxel_size=0.01,
                 text_unfreeze_layers=0, vision_unfreeze_layers=0):
        """Initialize layers.

        Args:
            text_unfreeze_layers (int): Number of last Transformer layers in the
                text encoder to finetune. ``0`` means fully frozen.
            vision_unfreeze_layers (int): Number of last stages in the visual
                backbone to finetune. ``0`` means fully frozen.
        """
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
        self.voxel_size = voxel_size

        # Visual encoder
        self.vision_backbone = TSPBackbone(in_channels=6)

        # Validate stage names to avoid silent freezing issues
        stage_names = [f"layer{i}" for i in range(4, 0, -1)]
        for name in stage_names:
            if not hasattr(self.vision_backbone, name):
                raise AttributeError(
                    f"TSPBackbone missing stage {name}; adjust vision_unfreeze_layers handling."
                )

        # Freeze all backbone params first
        for param in self.vision_backbone.parameters():
            param.requires_grad = False

        # Track trainable stages for BN handling later
        self._trainable_stages = []
        if vision_unfreeze_layers > 0:
            for name in stage_names[:vision_unfreeze_layers]:
                stage = getattr(self.vision_backbone, name)
                for param in stage.parameters():
                    param.requires_grad = True
                self._trainable_stages.append(stage)

        # Freeze BN running stats in frozen stages
        self._freeze_bn_in_frozen_stages()

        # Text encoder
        t_type = os.path.join(data_path, "roberta-base") if data_path else "roberta-base"
        try:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
            self.text_encoder = RobertaModel.from_pretrained(
                t_type, local_files_only=True, use_safetensors=False
            )
        except (OSError, ValueError):
            # Fall back to downloading weights if local files are unavailable
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            self.text_encoder = RobertaModel.from_pretrained(
                "roberta-base", use_safetensors=False
            )
        # Freeze all parameters then optionally unfreeze last ``text_unfreeze_layers`` blocks
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        if text_unfreeze_layers > 0:
            for layer in self.text_encoder.encoder.layer[-text_unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # self.neck = TR3DNeck()
        self.head = TSPHead(voxel_size=self.voxel_size)

    def _freeze_bn(self, module):
        """Set BatchNorm layers to eval mode and disable gradients."""
        FrozenBN = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            ME.MinkowskiBatchNorm,
            getattr(ME, "MinkowskiSyncBatchNorm", nn.Identity),
        )
        for m in module.modules():
            if isinstance(m, FrozenBN):
                m.eval()
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad_(False)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(False)

    def _unfreeze_bn(self, module):
        """Enable training mode and gradients for BatchNorm layers."""
        FrozenBN = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            ME.MinkowskiBatchNorm,
            getattr(ME, "MinkowskiSyncBatchNorm", nn.Identity),
        )
        for m in module.modules():
            if isinstance(m, FrozenBN):
                m.train()
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad_(True)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)

    def _freeze_bn_in_frozen_stages(self):
        """Freeze BN stats for all frozen stages."""
        self._freeze_bn(self.vision_backbone)
        for stage in self._trainable_stages:
            self._unfreeze_bn(stage)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # ensure frozen stages keep BN in eval
            self._freeze_bn_in_frozen_stages()
        return self
        
    
    # BRIEF forward.
    def forward(self, inputs, gt_bboxes=None, gt_labels=None, gt_all_bbox_new=None, 
                auxi_bbox=None, img_metas=None, epoch=None):
        """
        Forward pass - 改进版本，确保返回所有loss组件
        """
        # Vision and text encoding
        points = inputs['point_clouds']
        start_time = time.time()
        coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, 0:] if p.shape[1] > 3 else p[:, :3]) for p in points],
                device=points[0].device)        
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.vision_backbone(x)
        visual_time = time.time() - start_time
        
        # Text encoding
        start_time = time.time()
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'].device)
        
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        # attention_mask: 1 for valid tokens, 0 for padding
        text_attention_mask = tokenized.attention_mask.eq(0)
        text_time = time.time() - start_time
        times = {
            "visual_time": visual_time,
            "text_time": text_time
        }

        if not self.training:
            # Test mode
            bbox_list, head_time = self.head.forward_test(x, text_feats, text_attention_mask, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results, {'loss': 0.}, 0., times
        
        # Training mode - 获取head的loss
        head_losses = self.head.forward_train(
            x, text_feats, text_attention_mask, 
            gt_bboxes, gt_labels, gt_all_bbox_new, 
            auxi_bbox, img_metas
        )
        
        # 如果使用DETR风格的decoder，添加matching loss
        if hasattr(self, 'decoder'):
            # 这里应该调用compute_hungarian_loss
            # 但是看起来这个在head里已经处理了
            pass
        
        # 确保包含所有loss组件
        all_losses = {}
        for key, value in head_losses.items():
            if 'loss' in key.lower():
                all_losses[key] = value
        
        # 计算总loss（如果没有的话）
        if 'loss' not in all_losses:
            total_loss = sum(v for v in all_losses.values() if torch.is_tensor(v))
            all_losses['loss'] = total_loss
        
        return all_losses
    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
