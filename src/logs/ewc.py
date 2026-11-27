"""Pôle 2 - EWC (Elastic Weight Consolidation) sur les paramètres entraînables.

Fisher diagonale "empirique par échantillon" : un gradient par exemple, avec une
étiquette tirée dans la sortie du modèle. Pénalité : lam/2 * sum F * (theta - theta*)^2.
"""
import random

import torch
import torch.nn.functional as F


def trainable(model) -> dict:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def snapshot(model) -> dict:
    """Copie des paramètres entraînables (theta*)."""
    return {n: p.detach().clone() for n, p in trainable(model).items()}


def restore(model, state: dict) -> None:
    with torch.no_grad():
        for n, p in trainable(model).items():
            p.copy_(state[n])


def compute_fisher(model, tok, texts, n_samples=200, max_len=256, seed=0, device="cuda"):
    """Fisher diagonale sur n_samples textes (forward en fp32, batch de 1)."""
    rng = random.Random(seed)
    sample = rng.sample(list(texts), min(n_samples, len(texts)))
    params = trainable(model)
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    model.eval()
    for text in sample:
        enc = tok([text], truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        model.zero_grad(set_to_none=True)
        logits = model(**enc).logits.float()
        y = torch.multinomial(logits.softmax(-1).detach(), 1).squeeze(-1)
        F.cross_entropy(logits, y).backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    model.zero_grad(set_to_none=True)
    return {n: f / len(sample) for n, f in fisher.items()}


def ewc_penalty(model, fisher: dict, theta_star: dict, lam: float):
    first = next(iter(trainable(model).values()))
    total = torch.zeros((), device=first.device, dtype=torch.float32)
    for n, p in model.named_parameters():
        if n in fisher:
            total = total + (fisher[n] * (p - theta_star[n]) ** 2).sum()
    return lam / 2.0 * total