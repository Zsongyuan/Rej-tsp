TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 3333 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/DATA_ROOT/ \
    --val_freq 3 --batch_size 7 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=8 \
    --dataset sr3d --test_dataset sr3d \
    --detect_intermediate --joint_det \
    --log_dir ~/DATA_ROOT/output/logs \
    --lr_decay_epochs 30 40 \