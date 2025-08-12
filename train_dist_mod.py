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
"""Main script for language modulation."""

import os
import random
import numpy as np
import torch
import torch.distributed as dist

from main_utils import parse_option, BaseTrainTester
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator, RejectionGroundingEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths

from tqdm import tqdm
import datetime

import ipdb
st = ipdb.set_trace

import numpy as np

class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    # BRIEF Initialize dataset.
    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        
        if args.joint_det and (args.wo_obj_name is None or args.wo_obj_name == "None"):
            dataset_dict['scannet'] = 10
        elif args.joint_det:
            print("[INFO] --wo_obj_name is provided. Skipping default 'scannet' dataset loading for training.")
            
        print('Loading datasets:', sorted(list(dataset_dict.keys())))

        if args.eval:
            train_dataset = None
        else:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color, use_height=args.use_height,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=args.augment_det,
                wo_obj_name=args.wo_obj_name, 
                val_file_path=None, 
                rejection_start_epoch=args.rejection_start_epoch,
                start_epoch=args.start_epoch
            )
        
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            wo_obj_name=args.wo_obj_name, 
            val_file_path=args.val_file_path
        )
        return train_dataset, test_dataset

    # BRIEF Initialize the model.
    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            data_path = args.data_root,
            self_attend=args.self_attend,
            voxel_size = args.voxel_size
        )
        return model

    # BRIEF input data.
    @staticmethod
    def _get_inputs(batch_data):
        # 基础输入
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            'target_cat': batch_data['target_cat']
        }

        if 'is_negative' in batch_data:
            inputs['is_negative'] = batch_data['is_negative']
        
        return inputs


    # BRIEF only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.
        """
        # [Option] Object detection evaluation on ScanNet dataset.
        if args.test_dataset == 'scannet':      
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model,
                criterion, set_criterion, args
            )

        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        
        # <<<<<<<<<<<<<<<<<<<<<<<< START: 修改代码 >>>>>>>>>>>>>>>>>>>>>>>>
        # 根据是否使用拒绝机制，动态选择和初始化评估器
        if args.use_rejection and args.val_file_path:
            self.logger.info("Using RejectionGroundingEvaluator for comprehensive evaluation.")
            evaluator = RejectionGroundingEvaluator(
                iou_thresh=0.5,  # 您可以将其改为args参数或保持固定
                rejection_thresh=args.rejection_thresh,
                voxel_size=args.voxel_size
            )
        else:
            self.logger.info("Using original GroundingEvaluator for localization accuracy only.")
            evaluator = GroundingEvaluator(
                only_root=True, thresholds=[0.25, 0.5],
                topks=[1], prefixes=['3dcnn'],
                filter_non_gt_boxes=args.butd_cls,
                voxel_size=args.voxel_size
            )
        # <<<<<<<<<<<<<<<<<<<<<<<< END: 修改代码 >>>>>>>>>>>>>>>>>>>>>>>>

        # NOTE Main eval branch
        test_loader = tqdm(test_loader, desc=f"Epoch {epoch} Evaluation")
        inf_speeds, vis_back_speeds, text_back_speeds, fuiosn_speeds, head_speeds = [],[],[],[],[]
        for batch_idx, batch_data in enumerate(test_loader):
            # note forward and compute loss
            stat_dict, end_points, inf_speed, backbone_time, detail_time  = self._main_eval_branch(     
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            inf_speeds.append(inf_speed)
            vis_back_speeds.append(detail_time.get(0, 0.0))
            text_back_speeds.append(detail_time.get(1, 0.0))
            fuiosn_speeds.append(detail_time.get(2, 0.0))
            head_speeds.append(detail_time.get(3, 0.0))
            
            if evaluator is not None:
                # 评估器的evaluate方法会根据样本类型执行不同逻辑
                evaluator.evaluate(end_points, '3dcnn')      

        evaluator.synchronize_between_processes()
        
        if dist.get_rank() == 0:
            if evaluator is not None:
                self.logger.info(f'Evaluation results for Epoch [{epoch}]:')
                evaluator.print_stats() # 新旧评估器都有这个方法，可以统一调用
                
                # 保留原始的日志输出（只对GroundingEvaluator有意义）
                if isinstance(evaluator, GroundingEvaluator):
                    for t in evaluator.thresholds:
                        self.logger.info(''.join([
                            f"{'3dcnn'} Acc@{t:.2f}: ", f"Top-{1}: {evaluator.dets[('3dcnn', t, 1, 'bbf')] / max(evaluator.gts[('3dcnn', t, 1, 'bbf')], 1):.5f}"
                        ]))

        print('inf: ', np.array(inf_speeds).mean(),'vis_back_speeds: ', np.array(vis_back_speeds).mean(),
              'text_back_speeds: ', np.array(text_back_speeds).mean(),'fuiosn_speeds: ', np.array(fuiosn_speeds).mean(),
              'head_speeds: ', np.array(head_speeds).mean())

        return None
       
    # BRIEF Scannet detection evalution
    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader,
                               model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes += [
                f'{i}head_' for i in range(args.num_decoder_layers - 1)
            ]
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes = ['last_']
        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in args.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in args.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        # NOTE char span and token span.
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])  # 18+1（not mentioned）
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])  # 18 token span

        test_loader = tqdm(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            # note eval
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )

            # step score   contrast
            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores  # ([B, 256, 256])

            # step
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19] # ([B, 256, 19])
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls     # ([B, 256, 19])

            # step Parse predictions
            # for prefix in prefixes:
            prefix = 'last_'
            # pred
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )

        return None


if __name__ == '__main__':
    # huggingface
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    opt = parse_option()       
    
    # distributed 
    # torch.cuda.set_device(opt.local_rank)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)
