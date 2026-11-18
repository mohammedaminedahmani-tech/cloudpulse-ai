"""Étape 5 : comparaison DQN vs baselines sur le jour de TEST (jamais vu à l'entraînement)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.infra.baselines import make_static_policy, run_episode, threshold_policy
from src.infra.env import ClusterEnv, load_series

MODEL_PATH = Path("models/dqn_best/best_model.zip")
DOCS = Path("docs")


def dqn_policy(model):
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def trajectory(env: ClusterEnv, policy):
    """Rejoue un épisode et garde, à chaque pas, la demande et le nombre de serveurs."""
    obs, _ = env.reset(seed=0)
    servers, needed = [], []
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        servers.append(info["n_servers"])
        needed.append(max(info["cpu_demand"], info["mem_demand"]) / env.capacity)
        if terminated or truncated:
            return np.array(servers), np.array(needed)


def main(model=None):
    if model is None:
        from stable_baselines3 import DQN
        model = DQN.load(MODEL_PATH)

    cpu, mem = load_series("test")
    peak = int(np.ceil(max(cpu.max(), mem.max()) / 100))
    new_env = lambda: ClusterEnv(cpu, mem, random_start=False)

    policies = {
        "DQN": lambda env: dqn_policy(model),
        "Seuils (HPA)": lambda env: threshold_policy,
        f"Statique ({peak})": lambda env: make_static_policy(env, peak),
    }

    # 1) Tableau des métriques
    rows = {}
    for name, make in policies.items():
        env = new_env()
        rows[name] = run_episode(env, make(env))
    table = pd.DataFrame(rows).T
    print(table.to_string())
    DOCS.mkdir(exist_ok=True)
    table.to_csv(DOCS / "resultats_test.csv")

    # 2) Graphique : serveurs alloués vs serveurs nécessaires
    fig, ax = plt.subplots(figsize=(14, 4))
    hours = None
    for name in ["DQN", "Seuils (HPA)"]:
        env = new_env()
        servers, needed = trajectory(env, policies[name](env))
        hours = np.arange(len(servers)) * 5 / 60
        ax.step(hours, servers, where="post", label=name)
    ax.plot(hours, needed, color="black", linewidth=1, label="Serveurs nécessaires")
    ax.set_xlabel("Temps (heures) - jour de test")
    ax.set_ylabel("Nombre de serveurs")
    ax.set_title("Autoscaling sur le jour de test : DQN vs HPA")
    ax.legend()
    plt.tight_layout()
    plt.savefig(DOCS / "dqn_vs_hpa.png", dpi=150)
    print("\nGraphique sauvegardé dans", DOCS / "dqn_vs_hpa.png")


if __name__ == "__main__":
    main()