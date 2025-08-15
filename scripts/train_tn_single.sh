TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 12221 \
    train_dist_mod.py \
    \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/multimodal/Rej-tsp/ \
    --val_freq 1 --batch_size 2 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=16 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --lr_decay_epochs 116 131 \
    --augment_det \
    --use_rejection \
    --text_unfreeze_layers 0 \
    --vision_unfreeze_layers 0 \
    --rejection_loss_weight 0.1 \
    --wo_obj_name ~/multimodal/Rej-tsp/tns/train_mixed_36665_0.5.json \
    --val_file_path ~/multimodal/Rej-tsp/tns/val_mixed_9508_0.5.json \
    --log_dir ~/multimodal/Rej-tsp/ScanRefer/output/logs_rejection \
    --checkpoint_path ~/multimodal/Rej-tsp/checkpoints/ckpt_scanrefer.pth \
    --start_epoch 67 \
    --rejection_start_epoch 67

