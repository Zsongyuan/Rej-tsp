import numpy as np
import torch
import torch.nn.functional as F
from itertools import product


def _ensure_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=torch.float32)
    return x.float()


def fit_temperature(z, y, init_T: float = 1.0):
    """Fit a temperature scalar using L-BFGS.

    Args:
        z (array-like): Logits.
        y (array-like): Binary labels {0,1}.
        init_T (float): Initial temperature.

    Returns:
        float: Optimal temperature ``T``.
    """
    z = _ensure_tensor(z).view(-1)
    y = _ensure_tensor(y).view(-1)

    log_T = torch.tensor(np.log(max(init_T, 1e-2)), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = torch.clamp(torch.exp(log_T), 1e-2, 100.0)
        loss = F.binary_cross_entropy_with_logits(z / T, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = torch.clamp(torch.exp(log_T), 1e-2, 100.0).item()
    return T


def fit_platt(z, y, init_a: float = 1.0, init_b: float = 0.0):
    """Fit Platt scaling parameters ``a`` and ``b`` using L-BFGS."""
    z = _ensure_tensor(z).view(-1)
    y = _ensure_tensor(y).view(-1)
    params = torch.tensor([init_a, init_b], requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        a, b = params[0], params[1]
        logits = torch.clamp(a, -20.0, 20.0) * z + torch.clamp(b, -20.0, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    a = float(torch.clamp(params[0], -20.0, 20.0).item())
    b = float(torch.clamp(params[1], -20.0, 20.0).item())
    return a, b


def grid_search_thresholds(scores_det, iou, is_neg, scores_rej, tn_floor=0.98):
    """Grid search for detection and rejection thresholds.

    Args:
        scores_det (list[list[float]]): Calibrated detection probabilities per sample.
        iou (list[list[float]]): IoU for each candidate per sample.
        is_neg (list[int]): 1 if negative sample else 0.
        scores_rej (list[float]): Calibrated rejection probabilities per sample.
        tn_floor (float): Minimum TN rate.

    Returns:
        dict: ``{'tau_det': ..., 'tau_rej': ..., 'loc': ..., 'tn': ...}``
    """
    taus_det = np.arange(0.1, 0.91, 0.02)
    taus_rej = np.arange(0.3, 0.91, 0.05)

    best = None
    best_score = -np.inf

    n_pos = max(1, np.sum(1 - np.array(is_neg)))
    n_neg = max(1, np.sum(is_neg))

    for td, tr in product(taus_det, taus_rej):
        tp = tn = fp = fn = 0
        for s_det, ious, neg, s_rej in zip(scores_det, iou, is_neg, scores_rej):
            max_det = max(s_det) if len(s_det) else 0.0
            max_iou = ious[int(np.argmax(s_det))] if len(s_det) else 0.0
            if neg:
                if s_rej >= tr or max_det < td:
                    tn += 1
                else:
                    fp += 1
            else:
                if max_det >= td and s_rej < tr and max_iou >= 0.5:
                    tp += 1
                else:
                    fn += 1
        tn_rate = tn / n_neg
        loc = tp / n_pos
        if tn_rate >= tn_floor:
            score = loc
        else:
            score = loc - 8.0 * max(0.0, tn_floor - tn_rate)
        if score > best_score:
            best_score = score
            best = {"tau_det": float(td), "tau_rej": float(tr), "loc": loc, "tn": tn_rate}
    return best
