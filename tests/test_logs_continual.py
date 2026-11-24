import pytest
import torch

from src.logs import metrics as m
from src.logs.ewc import ewc_penalty, restore, snapshot


def test_penalite_nulle_a_theta_star():
    model = torch.nn.Linear(3, 2)
    theta = snapshot(model)
    fisher = {n: torch.ones_like(p) for n, p in theta.items()}
    assert ewc_penalty(model, fisher, theta, lam=10.0).item() == 0.0


def test_penalite_apres_deplacement():
    model = torch.nn.Linear(3, 2)
    theta = snapshot(model)
    fisher = {n: torch.ones_like(p) for n, p in theta.items()}
    with torch.no_grad():
        model.weight += 1.0          # 6 poids déplacés de 1
    # lam/2 * 6 * 1^2 = 30
    assert ewc_penalty(model, fisher, theta, lam=10.0).item() == pytest.approx(30.0)
    restore(model, theta)
    assert ewc_penalty(model, fisher, theta, lam=10.0).item() == 0.0


def test_metriques_matrice_connue():
    R = [[0.9, 0.1], [0.5, 0.8]]
    assert m.forgetting(R) == pytest.approx(0.4)
    assert m.bwt(R) == pytest.approx(-0.4)
    assert m.average_final(R) == pytest.approx(0.65)