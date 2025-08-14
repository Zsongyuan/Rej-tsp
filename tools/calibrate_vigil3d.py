import argparse
import json
import os
import pickle
from collections import defaultdict
import numpy as np

from src.calibration import fit_platt, fit_temperature, grid_search_thresholds


def load_logits(det_path, rej_path):
    with open(det_path, 'rb') as f:
        det_records = pickle.load(f)
    with open(rej_path, 'rb') as f:
        rej_records = pickle.load(f)
    det_map = defaultdict(list)
    for r in det_records:
        det_map[r['sample_id']].append((r['z_det'], r['iou'], r['is_negative']))
    rej_map = {r['sample_id']: (r['z_rej_max'], r['is_negative']) for r in rej_records}
    return det_map, rej_map


def compute_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.any():
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def auc_score(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


def split_folds(sample_ids, k=5):
    scenes = list(sample_ids)
    np.random.shuffle(scenes)
    folds = [scenes[i::k] for i in range(k)]
    return folds


def fit_calibrators(det_map, rej_map, train_ids, method):
    z_det, y_det = [], []
    for sid in train_ids:
        for z, iou, neg in det_map[sid]:
            z_det.append(z)
            y_det.append(1 if iou >= 0.5 else 0)
    z_rej, y_rej = [], []
    for sid in train_ids:
        z, neg = rej_map[sid]
        z_rej.append(z)
        y_rej.append(neg)
    if method == 'temp':
        T_det = fit_temperature(z_det, y_det)
        T_rej = fit_temperature(z_rej, y_rej)
        det_cfg = {'type': 'temp', 'T': T_det}
        rej_cfg = {'type': 'temp', 'T': T_rej}
    else:
        a_det, b_det = fit_platt(z_det, y_det)
        a_rej, b_rej = fit_platt(z_rej, y_rej)
        det_cfg = {'type': 'platt', 'a': a_det, 'b': b_det}
        rej_cfg = {'type': 'platt', 'a': a_rej, 'b': b_rej}
    return {'detector': det_cfg, 'rejector': rej_cfg}


def apply_calibrator(z, cfg):
    if cfg['type'] == 'platt':
        return 1 / (1 + np.exp(-(cfg['a'] * z + cfg['b'])))
    else:
        return 1 / (1 + np.exp(-(z / cfg['T'])))


def evaluate_fold(det_map, rej_map, val_ids, calib, tn_floor):
    scores_det, ious, neg_flags, scores_rej = [], [], [], []
    det_flat, labels_flat = [], []
    for sid in val_ids:
        dlist = det_map[sid]
        z = np.array([d[0] for d in dlist])
        iou = np.array([d[1] for d in dlist])
        neg = dlist[0][2]
        p = apply_calibrator(z, calib['detector'])
        scores_det.append(p.tolist())
        ious.append(iou.tolist())
        neg_flags.append(int(neg))
        det_flat.extend(p.tolist())
        labels_flat.extend([1 if v >= 0.5 else 0 for v in iou])
        z_rej, n = rej_map[sid]
        p_rej = apply_calibrator(z_rej, calib['rejector'])
        scores_rej.append(p_rej)
    auc = auc_score(labels_flat, det_flat)
    ece = compute_ece(det_flat, labels_flat)
    best = grid_search_thresholds(scores_det, ious, neg_flags, scores_rej, tn_floor)
    return auc, ece, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_logits', required=True)
    parser.add_argument('--rej_logits', required=True)
    parser.add_argument('--method', default='platt', choices=['platt', 'temp'])
    parser.add_argument('--tn_floor', type=float, default=0.98)
    parser.add_argument('--out_dir', default='output/calib')
    args = parser.parse_args()

    det_map, rej_map = load_logits(args.det_logits, args.rej_logits)
    sample_ids = list(det_map.keys())
    folds = split_folds(sample_ids, k=5)

    aucs, eces, locs, tns = [], [], [], []
    for k in range(5):
        val_ids = folds[k]
        train_ids = [sid for i, f in enumerate(folds) if i != k for sid in f]
        calib = fit_calibrators(det_map, rej_map, train_ids, args.method)
        auc, ece, metrics = evaluate_fold(det_map, rej_map, val_ids, calib, args.tn_floor)
        aucs.append(auc)
        eces.append(ece)
        locs.append(metrics['loc'])
        tns.append(metrics['tn'])
        print(f"Fold {k}: AUC={auc:.3f} ECE={ece:.3f} Loc@0.5={metrics['loc']:.3f} TN={metrics['tn']:.3f}")

    print("CV Averages -> AUC={:.3f} ECE={:.3f} Loc@0.5={:.3f} TN={:.3f}".format(
        np.nanmean(aucs), np.mean(eces), np.mean(locs), np.mean(tns)))

    # fit on full data
    calib = fit_calibrators(det_map, rej_map, sample_ids, args.method)
    scores_det = []
    ious = []
    neg_flags = []
    scores_rej = []
    for sid in sample_ids:
        dlist = det_map[sid]
        z = np.array([d[0] for d in dlist])
        iou = np.array([d[1] for d in dlist])
        neg = dlist[0][2]
        p = apply_calibrator(z, calib['detector'])
        scores_det.append(p.tolist())
        ious.append(iou.tolist())
        neg_flags.append(int(neg))
        z_rej, n = rej_map[sid]
        scores_rej.append(apply_calibrator(z_rej, calib['rejector']))
    best = grid_search_thresholds(scores_det, ious, neg_flags, scores_rej, args.tn_floor)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'vigil3d_calibrator.json'), 'w') as f:
        json.dump(calib, f, indent=2)
    best_out = {"tau_det": best['tau_det'], "tau_rej": best['tau_rej'], "tn_floor": args.tn_floor}
    with open(os.path.join(args.out_dir, 'vigil3d_thresholds.json'), 'w') as f:
        json.dump(best_out, f, indent=2)
    print("Saved calibrator and thresholds to", args.out_dir)


if __name__ == '__main__':
    main()