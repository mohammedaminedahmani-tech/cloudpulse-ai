"""Pôle 2 - Calibration par température et ECE (CPU, testable sans GPU)."""
import numpy as np
import torch
import torch.nn.functional as F


def softmax(logits, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, labels, n_bins: int = 10) -> float:
    """Expected Calibration Error : écart moyen |précision - confiance| par tranche."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits, labels) -> float:
    """Cherche T > 0 qui minimise la perte (NLL) de softmax(logits / T)."""
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))