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
"""Shared utilities for all main scripts."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm
from get_gt import get_gt

def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')
    parser.add_argument('--voxel_size', default=0.01, type=float)
    parser.add_argument('--dim_is_radius', action='store_true',
                        help='If model predicts half-dimensions (radius).')
    parser.add_argument('--axis_perm', type=int, nargs=3, default=[0, 1, 2],
                        help='Permutation of axes from local to world.')
    parser.add_argument('--axis_sign', type=int, nargs=3, default=[1, 1, 1],
                        help='Sign of each axis when mapping to world.')
    parser.add_argument('--use_scene_offset', action='store_true',
                        help='Apply per-scene offset to predictions.')
    parser.add_argument('--offset_keys', type=str, nargs='+',
                        default=['scene_offset', 'origin', 'pc_min', 'shift', 'scene_shift'],
                        help='Keys to search for scene offset in end_points.')
    parser.add_argument('--gt_in_world', action='store_true', default=True,
                        help='Ground truth boxes already in world coordinates.')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)    # 6
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['sr3d'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='sr3d')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')     # color
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--wo_obj_name', default='None')    # grounding without object name
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--keep_trans_lr", default=4e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument("--box_select_lr", default=4e-4, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')  # note
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)    # pointnet checkpoint
    parser.add_argument('--reduce_lr', action='store_true')

    # rejection
    parser.add_argument('--use_rejection', action='store_true',
                        help='Enable training with instruction rejection.')
    parser.add_argument('--rejection_loss_weight', type=float, default=1.0,
                        help='Weight for the rejection loss.')
    parser.add_argument('--positive_sample_ratio', type=float, default=0.5,
                        help='Ratio of positive samples in a batch when using rejection training.')
    
    parser.add_argument('--val_file_path', type=str, default=None,
                        help='Path to the mixed validation json file.')
    parser.add_argument('--rejection_thresh', type=float, default=0.5,
                        help='Confidence threshold to count an instruction as rejected.')
    parser.add_argument('--rejection_start_epoch', type=int, default=1,
                        help='Epoch to start training with rejection samples and loss.')

    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train

    return args

# BRIEF load checkpoint.
def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    model.load_state_dict(checkpoint['model'], strict=False)
    # if not args.eval and not args.reduce_lr:
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()


# BRIEF save model.
def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class BaseTrainTester:
    def __init__(self, args):
        """Initialize with TensorBoard support"""
        name = args.log_dir.split('/')[-1] 
        
        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )

        # Initialize TensorBoard
        if dist.get_rank() == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(args.log_dir)
        else:
            self.tb_writer = None

        # Save config
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))
    
    def log_losses(self, losses, step, prefix='train'):
        """Log losses to TensorBoard"""
        if self.tb_writer is not None:
            for key, value in losses.items():
                if 'loss' in key.lower():
                    if hasattr(value, 'item'):
                        value = value.item()
                    self.tb_writer.add_scalar(f'{prefix}/{key}', value, step)

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset


    # BRIEF dataloader.
    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)

        if args.eval:
            train_loader = None
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,      # TODO 
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g
            )
        
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        losses = ['boxes', 'labels']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        param_dicts = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "keep_trans" not in n and "text_encoder" not in n
                    and "select" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "keep_trans" in n and p.requires_grad
                ],
                "lr": args.keep_trans_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "select" in n and p.requires_grad
                ],
                "lr": args.box_select_lr
            }
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer


    # BRIEF main training/testing
    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if not args.eval:
            scheduler = get_scheduler(optimizer, len(train_loader), args)
        else:
            scheduler = None
        
        # Move model to devices
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # synBN
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            else:
                model = model.cuda()

        # note Distributed Data-Parallel Training (DDP)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False  , find_unused_parameters=True
        )

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
        
        # ##############################################
        # NOTE [eval-only] Just eval and end execution #
        # ##############################################
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            return

        # ##############################
        # NOTE Training and Validation #
        # ##############################
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )
            
            # log
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, '
                'lr_tran {:.5f}, '
                'lr_text {:.5f}, '
                'lr_select {:.5f}, '.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr'],
                    optimizer.param_groups[2]['lr'],
                    optimizer.param_groups[3]['lr']
                )
            )

            # save model and validate
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                
                # validate *
                print("Test evaluation.......")
                self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )

        # Training is over
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        """获取模型输入数据"""
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
        }

        # Record per-scene offset so predictions can be restored
        offset = batch_data['point_clouds'][:, :, :3].min(dim=1)[0]
        inputs['scene_offset'] = offset
        batch_data['scene_offset'] = offset
        
        # 添加可选字段
        if 'target_cat' in batch_data:
            inputs['target_cat'] = batch_data['target_cat']
        
        if 'is_negative' in batch_data:
            inputs['is_negative'] = batch_data['is_negative']
        
        return inputs
        
    def format_loss_log(losses_dict, prefix=""):
        """
        格式化loss字典为可读的字符串
        Args:
            losses_dict: 包含各种loss的字典
            prefix: 输出前缀
        Returns:
            格式化的字符串
        """
        loss_items = []
        
        # 首先添加总loss
        if 'loss' in losses_dict:
            loss_items.append(f"Total: {losses_dict['loss']:.4f}")
        
        # 然后添加各个组件loss
        for key in sorted(losses_dict.keys()):
            if key != 'loss' and 'loss' in key.lower():
                value = losses_dict[key]
                if hasattr(value, 'item'):
                    value = value.item()
                
                # 简化名称
                name = key.replace('_loss', '').replace('loss_', '').capitalize()
                loss_items.append(f"{name}: {value:.4f}")
        
        return f"{prefix}{' | '.join(loss_items)}"
    @staticmethod
    def _get_inputs_contra(batch_data):
        gt_labels = batch_data['sem_cls_label']
        gt_center = batch_data['center_label'][:, :, 0:3]
        gt_size = batch_data['size_gts']
        gt_bbox = torch.cat([gt_center, gt_size], dim=-1) 
        positive_map = batch_data['positive_map']               # main obj.
        modify_positive_map = batch_data['modify_positive_map'] # attribute(modify)
        pron_positive_map = batch_data['pron_positive_map']     # pron
        other_entity_map = batch_data['other_entity_map']       # other(auxi)
        rel_positive_map = batch_data['rel_positive_map']       # relation
        box_label_mask = batch_data['box_label_mask'] 
        target = [
            {
                "boxes": gt_bbox[b, box_label_mask[b].bool()],
                "positive_map": positive_map[b, box_label_mask[b].bool()],
                "modify_positive_map": modify_positive_map[b, box_label_mask[b].bool()],
                "pron_positive_map": pron_positive_map[b, box_label_mask[b].bool()],
                "other_entity_map": other_entity_map[b, box_label_mask[b].bool()],
                "rel_positive_map": rel_positive_map[b, box_label_mask[b].bool()]
            }
            for b in range(gt_labels.shape[0])
        ]       
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            'target':target
        }

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk,
            rejection_loss_weight=args.rejection_loss_weight if args.use_rejection else 0.0
        )
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict


    # BRIEF Training
    def train_one_epoch(self, epoch, train_loader, model,
                                criterion, set_criterion,
                                optimizer, scheduler, args):
        """完整的训练函数，显示所有loss组件"""
        model.train()
        
        # 初始化所有loss的meter
        meters = {
            'total': AverageMeter(),
            'bbox': AverageMeter(),
            'cls': AverageMeter(),
            'keep': AverageMeter(),
            'com': AverageMeter(),
            'ce': AverageMeter(),
            'giou': AverageMeter(),
            'sem_align': AverageMeter(),
            'rejection': AverageMeter(),  # 添加rejection loss meter
            'query_points': AverageMeter(),
        }
        
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 准备数据
            gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, img_metas = get_gt(batch_data)
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)
            
            # Forward pass
            losses = model(inputs, gt_bboxes_3d, gt_labels_3d, 
                        gt_all_bbox_new, auxi_bbox, img_metas, epoch)
            
            # 获取总loss用于backward
            total_loss = losses.get('loss')
            if total_loss is None:
                # 如果没有总loss，计算所有loss的和
                total_loss = sum(v for k, v in losses.items() 
                            if 'loss' in k and torch.is_tensor(v))
                losses['loss'] = total_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            grad_norm = 0
            if args.clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
            
            optimizer.step()
            scheduler.step()
            
            # 更新meters
            batch_size = len(batch_data['scan_ids'])
            meters['total'].update(total_loss.item(), batch_size)
            
            # 更新各个loss组件的meter
            loss_mapping = {
                'bbox_loss': 'bbox',
                'cls_loss': 'cls',
                'keep_loss': 'keep',
                'com_loss': 'com',
                'loss_ce': 'ce',
                'loss_giou': 'giou',
                'loss_sem_align': 'sem_align',
                'loss_rejection': 'rejection',  # 添加rejection loss
                'rejection_loss': 'rejection',  # 两种可能的命名
                'loss_query_points': 'query_points',
                'query_points_generation_loss': 'query_points',
            }
            
            for loss_key, meter_key in loss_mapping.items():
                if loss_key in losses:
                    value = losses[loss_key]
                    if torch.is_tensor(value):
                        value = value.item()
                    meters[meter_key].update(value, batch_size)
            
            # 更新进度条显示
            postfix = {
                'Loss': f'{meters["total"].avg:.4f}',
                'BBox': f'{meters["bbox"].avg:.4f}',
                'Cls': f'{meters["cls"].avg:.4f}',
                'Rej': f'{meters["rejection"].avg:.4f}',  # 显示rejection loss
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            }
            train_loader.set_postfix(postfix)
            
            # 定期详细日志
            if (batch_idx + 1) % args.print_freq == 0:
                log_msg = (
                    f'Train [{epoch}][{batch_idx+1}/{len(train_loader)}]: '
                    f'Loss {meters["total"].avg:.4f} '
                    f'(BBox: {meters["bbox"].avg:.4f}, '
                    f'Cls: {meters["cls"].avg:.4f}, '
                    f'Keep: {meters["keep"].avg:.4f}, '
                    f'Com: {meters["com"].avg:.4f}, '
                    f'CE: {meters["ce"].avg:.4f}, '
                    f'GIoU: {meters["giou"].avg:.4f}, '
                    f'Sem: {meters["sem_align"].avg:.4f}, '
                    f'Rej: {meters["rejection"].avg:.4f}, '  # 添加rejection loss
                    f'QP: {meters["query_points"].avg:.4f}) '
                    f'LR: {optimizer.param_groups[0]["lr"]:.6f} '
                    f'Grad: {grad_norm:.4f}'
                )
                self.logger.info(log_msg)
                
                # TensorBoard logging
                if self.tb_writer is not None:
                    global_step = epoch * len(train_loader) + batch_idx
                    for name, meter in meters.items():
                        self.tb_writer.add_scalar(
                            f'train/{name}_loss', meter.avg, global_step
                        )
                    self.tb_writer.add_scalar(
                        'train/grad_norm', grad_norm, global_step
                    )
                    self.tb_writer.add_scalar(
                        'train/learning_rate', 
                        optimizer.param_groups[0]["lr"], 
                        global_step
                    )
        
        # Epoch结束总结
        summary = (
            f'\nEpoch {epoch} Training Summary:\n'
            f'  Total Loss: {meters["total"].avg:.4f}\n'
            f'  Detection Losses:\n'
            f'    - BBox Loss: {meters["bbox"].avg:.4f}\n'
            f'    - Class Loss: {meters["cls"].avg:.4f}\n'
            f'    - GIoU Loss: {meters["giou"].avg:.4f}\n'
            f'  Grounding Losses:\n'
            f'    - CE Loss: {meters["ce"].avg:.4f}\n'
            f'    - Semantic Align: {meters["sem_align"].avg:.4f}\n'
            f'    - Rejection Loss: {meters["rejection"].avg:.4f}\n'  # 添加rejection loss
            f'  Auxiliary Losses:\n'
            f'    - Keep Loss: {meters["keep"].avg:.4f}\n'
            f'    - Com Loss: {meters["com"].avg:.4f}\n'
            f'    - Query Points: {meters["query_points"].avg:.4f}\n'
        )
        self.logger.info(summary)
        
        return meters

    # BRIEF eval 
    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Move to GPU
        gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, img_metas = get_gt(batch_data)
        batch_data = self._to_gpu(batch_data)
        # inputs = self._get_inputs_contra(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False
            
        
        # STEP Forward pass
        start_time = time.time()
        bbox_results, losses, backbone_time, trans_time = model(inputs, gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, img_metas=img_metas)
        end_time = time.time()
        inf_time = end_time - start_time
        
        end_points = {'bbox_results': bbox_results, 'gt_bboxes_3d':gt_bboxes_3d}
        # STEP Compute loss
        for key in batch_data:
            assert (key not in end_points)
            end_points[key] = batch_data[key]

        stat_dict = self._accumulate_stats(stat_dict, losses)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key
            ]))
        return stat_dict, end_points, inf_time, backbone_time, trans_time

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
    
class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RejectionMonitor:
    """监控rejection机制的训练效果"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos_losses = []
        self.neg_losses = []
        self.pos_confidences = []
        self.neg_confidences = []
    
    def update(self, losses, is_negative, predictions):
        """更新统计"""
        if is_negative:
            self.neg_losses.append(losses['loss_rejection'].item())
            # 记录负样本的最高置信度
            max_conf = torch.max(torch.sigmoid(predictions)).item()
            self.neg_confidences.append(max_conf)
        else:
            self.pos_losses.append(losses.get('loss_bbox', 0).item())
            # 记录正样本的最高置信度
            max_conf = torch.max(torch.sigmoid(predictions)).item()
            self.pos_confidences.append(max_conf)
    
    def get_stats(self):
        """获取统计信息"""
        stats = {
            'avg_pos_conf': np.mean(self.pos_confidences) if self.pos_confidences else 0,
            'avg_neg_conf': np.mean(self.neg_confidences) if self.neg_confidences else 0,
            'avg_pos_loss': np.mean(self.pos_losses) if self.pos_losses else 0,
            'avg_neg_loss': np.mean(self.neg_losses) if self.neg_losses else 0,
            'num_pos': len(self.pos_losses),
            'num_neg': len(self.neg_losses),
        }
        return stats
    
    def log_stats(self, logger, epoch):
        """记录统计信息"""
        stats = self.get_stats()
        logger.info(
            f'Rejection Stats - Epoch {epoch}:\n'
            f'  Positive Samples: {stats["num_pos"]}\n'
            f'    - Avg Confidence: {stats["avg_pos_conf"]:.4f}\n'
            f'    - Avg Loss: {stats["avg_pos_loss"]:.4f}\n'
            f'  Negative Samples: {stats["num_neg"]}\n'
            f'    - Avg Confidence: {stats["avg_neg_conf"]:.4f}\n'
            f'    - Avg Rejection Loss: {stats["avg_neg_loss"]:.4f}\n'
            f'  Confidence Gap: {stats["avg_pos_conf"] - stats["avg_neg_conf"]:.4f}'
        )