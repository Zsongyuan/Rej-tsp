#!/bin/bash
set -e

# Generate logits on dev split
bash scripts/test_vigil3d_single.sh --dump_calib --calib_topk 5 "$@"

# Fit calibrators and thresholds
python tools/calibrate_vigil3d.py \
  --det_logits output/calib/vigil3d_det_logits.pkl \
  --rej_logits output/calib/vigil3d_rej_logits.pkl \
  --method platt --tn_floor 0.98 \
  --out_dir output/calib/