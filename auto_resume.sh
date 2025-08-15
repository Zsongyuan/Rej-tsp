#!/usr/bin/env bash
# 自动监控并断点重启训练（基于最新 ckpt_epoch_xx.pth）
# 运行位置：建议在 ~/zhu/Rej-tsp 下执行：  bash auto_resume.sh

set -uo pipefail

# =============== 可调参数 ===============
# ckpt 搜索根目录（会递归子目录）
CKPT_ROOT="./ScanRefer/output/logs_rejection/scanrefer"

# 异常退出后，等待多少秒再重启
RESTART_DELAY=10

# 是否从下一轮开始（0=用 ckpt 的 epoch；1=从 epoch+1 开始）
RESUME_NEXT_EPOCH=1

# 固定端口（与原命令一致）。如遇端口占用，可手动改一个。
MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)}


# =============== 查找最新 ckpt ===============
find_latest_ckpt() {
  local root="$1"
  # 收集所有 ckpt_epoch_*.pth
  mapfile -t files < <(find "$root" -type f -name 'ckpt_epoch_*.pth' 2>/dev/null | sort)
  if [ ${#files[@]} -eq 0 ]; then
    return 1
  fi

  local best_epoch=-1
  local best_file=""

  for f in "${files[@]}"; do
    local base
    base="$(basename "$f")"
    if [[ "$base" =~ ckpt_epoch_([0-9]+)\.pth$ ]]; then
      local ep="${BASH_REMATCH[1]}"
      if (( ep > best_epoch )); then
        best_epoch=$ep
        best_file="$f"
      fi
    fi
  done

  if (( best_epoch < 0 )); then
    return 1
  fi

  # 输出：文件绝对路径;epoch
  local abs
  abs="$(readlink -f "$best_file")"
  echo "${abs};${best_epoch}"
  return 0
}

# =============== 构建并运行训练命令 ===============
run_training() {
  local ckpt_path="$1"
  local start_epoch="$2"

  # 日志目录
  mkdir -p runs
  local ts
  ts="$(date +%F_%H-%M-%S)"
  local logfile="runs/train_${ts}_ep${start_epoch}.log"

  echo "[INFO] 使用权重: ${ckpt_path}"
  echo "[INFO] start_epoch: ${start_epoch}"
  echo "[INFO] 日志: ${logfile}"

  # 原样使用你的分布式命令（仅替换 --checkpoint_path 与 --start_epoch）
  set +e
  TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch \
      --nproc_per_node 4 --master_port "${MASTER_PORT}" \
      train_dist_mod.py \
      \
      --use_color \
      --weight_decay 0.0005 \
      --data_root ~/zhu/Rej-tsp/ \
      --val_freq 1 --batch_size 5 --save_freq 1 --print_freq 500 \
      --lr=5e-5 --keep_trans_lr=5e-5 --voxel_size=0.01 --num_workers=16 \
      --dataset scanrefer --test_dataset scanrefer \
      --detect_intermediate --joint_det \
      --lr_decay_epochs 116 143 \
      --augment_det \
      --use_rejection \
      --rejection_loss_weight 0.1 \
      --wo_obj_name ~/zhu/Rej-tsp/tns/train_mixed_36665.json \
      --val_file_path ~/zhu/Rej-tsp/tns/val_mixed.json \
      --log_dir ~/zhu/Rej-tsp/ScanRefer/output/logs_rejection \
      --checkpoint_path "${ckpt_path}" \
      --start_epoch "${start_epoch}" \
      --rejection_start_epoch 82 \
      2>&1 | tee -a "${logfile}"
  local code=${PIPESTATUS[0]}
  set -e
  return ${code}
}

# 友好退出
trap 'echo; echo "[INFO] 捕获到中断信号，退出监控。"; exit 0' INT TERM

# =============== 主循环：监控 + 自动重启 ===============
while true; do
  LATEST=$(find_latest_ckpt "$CKPT_ROOT") || {
    echo "[ERROR] 没有找到 ckpt_epoch_*.pth，请确认目录：$CKPT_ROOT"
    exit 1
  }
  IFS=';' read -r CKPT_FILE EPOCH <<< "$LATEST"

  START_EPOCH="$EPOCH"
  if (( RESUME_NEXT_EPOCH == 1 )); then
    START_EPOCH=$((EPOCH + 1))
  fi

  run_training "$CKPT_FILE" "$START_EPOCH"
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] 训练进程正常结束（exit=0），停止监控。"
    break
  else
    echo "[WARN] 训练异常退出（exit=${EXIT_CODE}）。${RESTART_DELAY}s 后自动重启并使用最新权重……"
    sleep "${RESTART_DELAY}"
  fi
done

