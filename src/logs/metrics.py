"""Pôle 2 - Métriques d'apprentissage continu.

R[i][j] = F1 sur la tâche j après avoir appris la tâche i (matrice T x T).
"""
import numpy as np


def average_final(R) -> float:
    """Moyenne des F1 sur toutes les tâches après la dernière."""
    return float(np.mean(np.asarray(R)[-1]))


def forgetting(R) -> float:
    """Oubli moyen : max_i R[i][j] - R[T-1][j], pour j < T-1."""
    R = np.asarray(R)
    T = len(R)
    return float(np.mean([R[:T - 1, j].max() - R[T - 1, j] for j in range(T - 1)]))


def bwt(R) -> float:
    """Transfert arrière : moyenne de R[T-1][j] - R[j][j], pour j < T-1."""
    R = np.asarray(R)
    T = len(R)
    return float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))