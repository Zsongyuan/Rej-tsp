import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from calibration import fit_platt, fit_temperature


def load_logits(det_path):
    with open(det_path, 'rb') as f:
        det_records = pickle.load(f)
    det_map = defaultdict(list)
    for r in det_records:
        det_map[r['sample_id']].append((r['z_det'], r['iou']))
    return det_map


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


def fit_calibrator(det_map, train_ids, method):
    z_det, y_det = [], []
    for sid in train_ids:
        for z, iou in det_map[sid]:
            z_det.append(z)
            y_det.append(1 if iou >= 0.5 else 0)
    if method == 'temp':
        T_det = fit_temperature(z_det, y_det)
        det_cfg = {'type': 'temp', 'T': T_det}
    else:
        a_det, b_det = fit_platt(z_det, y_det)
        det_cfg = {'type': 'platt', 'a': a_det, 'b': b_det}
    return {'detector': det_cfg}


def apply_calibrator(z, cfg):
    if cfg['type'] == 'platt':
        return 1 / (1 + np.exp(-(cfg['a'] * z + cfg['b'])) )
    else:
        return 1 / (1 + np.exp(-(z / cfg['T'])))


def grid_search_det_threshold(scores_det, ious):
    taus = np.arange(0.3, 0.91, 0.05)
    best = {'tau_det': 0.5, 'loc': 0.0}
    for td in taus:
        locs = []
        for s_det, iou in zip(scores_det, ious):
            max_det = np.max(s_det)
            max_iou = np.max(iou)
            locs.append(1 if (max_det >= td and max_iou >= 0.5) else 0)
        loc = np.mean(locs)
        if loc > best['loc']:
            best = {'tau_det': float(td), 'loc': float(loc)}
    return best


def evaluate_fold(det_map, val_ids, calib):
    scores_det, ious = [], []
    det_flat, labels_flat = [], []
    for sid in val_ids:
        dlist = det_map[sid]
        z = np.array([d[0] for d in dlist])
        iou = np.array([d[1] for d in dlist])
        p = apply_calibrator(z, calib['detector'])
        scores_det.append(p.tolist())
        ious.append(iou.tolist())
        det_flat.extend(p.tolist())
        labels_flat.extend([1 if v >= 0.5 else 0 for v in iou])
    auc = auc_score(labels_flat, det_flat)
    ece = compute_ece(det_flat, labels_flat)
    best = grid_search_det_threshold(scores_det, ious)
    return auc, ece, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_logits', required=True)
    parser.add_argument('--method', default='platt', choices=['platt', 'temp'])
    parser.add_argument('--out_dir', default='output/calib')
    args = parser.parse_args()

    det_map = load_logits(args.det_logits)
    sample_ids = list(det_map.keys())
    folds = split_folds(sample_ids, k=5)

    aucs, eces, locs = [], [], []
    for k in range(5):
        val_ids = folds[k]
        train_ids = [sid for i, f in enumerate(folds) if i != k for sid in f]
        calib = fit_calibrator(det_map, train_ids, args.method)
        auc, ece, metrics = evaluate_fold(det_map, val_ids, calib)
        aucs.append(auc)
        eces.append(ece)
        locs.append(metrics['loc'])
        print(f"Fold {k}: AUC={auc:.3f} ECE={ece:.3f} Loc@0.5={metrics['loc']:.3f}")

    print("CV Averages -> AUC={:.3f} ECE={:.3f} Loc@0.5={:.3f}".format(
        np.nanmean(aucs), np.mean(eces), np.mean(locs)))

    calib = fit_calibrator(det_map, sample_ids, args.method)
    scores_det, ious = [], []
    for sid in sample_ids:
        dlist = det_map[sid]
        z = np.array([d[0] for d in dlist])
        iou = np.array([d[1] for d in dlist])
        p = apply_calibrator(z, calib['detector'])
        scores_det.append(p.tolist())
        ious.append(iou.tolist())
    best = grid_search_det_threshold(scores_det, ious)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'vigil3d_calibrator.json'), 'w') as f:
        json.dump(calib, f, indent=2)
    with open(os.path.join(args.out_dir, 'vigil3d_thresholds.json'), 'w') as f:
        json.dump({'tau_det': best['tau_det']}, f, indent=2)
    print('Saved calibrator and thresholds to', args.out_dir)


if __name__ == '__main__':
    main()