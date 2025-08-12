TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 2222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/multimodal/Rej-tsp/ \
    --val_freq 3 --batch_size 6 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=1.0 --num_workers=8 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate \
    --log_dir ~/multimodal/Rej-tsp/output/logs/test \
    --lr_decay_epochs 50 75 \
    --augment_det \
    --checkpoint_path ~/multimodal/Rej-tsp/checkpoints/ckpt_scanrefer.pth \
    --use_scene_offset \
    --axis_perm 0 1 2 --axis_sign 1 1 1 \
    --debug \
    --eval