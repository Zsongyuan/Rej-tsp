import numpy as np
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
                 self_attend=True, voxel_size=0.01):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
        self.voxel_size = voxel_size

        # Visual encoder
        self.vision_backbone = TSPBackbone(in_channels=6)
        
        # Text encoder
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        # self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True, use_safetensors=False)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )       
        
        # self.neck = TR3DNeck()
        self.head = TSPHead(voxel_size=self.voxel_size)
        
    
    # BRIEF forward.
    def forward(self, inputs, gt_bboxes=None, gt_labels=None, gt_all_bbox_new=None, auxi_bbox=None, img_metas=None, epoch=None):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
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
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_time = time.time() - start_time
        
        if not self.training:
            start_time = time.time()
            bbox_list, head_time = self.head.forward_test(x, text_feats, text_attention_mask, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            fusion_time = time.time() - start_time
            return bbox_results, {'loss':0.}, 0., [visual_time,text_time,fusion_time-head_time,head_time]
        losses = self.head.forward_train(x,text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas)
        losses.update({'loss':sum(value for key, value in losses.items() if '_loss' in key)})
        return losses
    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
