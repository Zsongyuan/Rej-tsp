TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 2222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/multimodal/Rej-tsp/ \
    --val_freq 3 --batch_size 8 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=8 \
    --dataset vigil3d --test_dataset vigil3d \
    --detect_intermediate \
    --log_dir ~/multimodal/Rej-tsp/output/logs/test \
    --lr_decay_epochs 50 75 \
    --augment_det \
    --checkpoint_path ~/multimodal/Rej-tsp/checkpoints/ckpt_epoch_97.pth \
    --eval 