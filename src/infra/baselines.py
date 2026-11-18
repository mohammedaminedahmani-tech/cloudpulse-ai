"""Étape 3 : politiques de référence (baselines) et fonction d'évaluation commune."""
import numpy as np

from src.infra.env import ClusterEnv, load_series

# Index des actions dans ClusterEnv.ACTIONS = [-5, -1, 0, +1, +5]
MINUS5, MINUS1, KEEP, PLUS1, PLUS5 = range(5)


def threshold_policy(obs, high=0.8, very_high=0.95, low=0.5):
    """Imite l'autoscaler de Kubernetes (HPA) : on réagit à l'utilisation ACTUELLE."""
    util = max(obs[0], obs[1])  # la ressource la plus chargée (CPU ou mémoire)
    if util > very_high:
        return PLUS5
    if util > high:
        return PLUS1
    if util < low:
        return MINUS1
    return KEEP


def make_static_policy(env: ClusterEnv, target: int):
    """Sur-provisionnement : on fixe un nombre de serveurs et on n'y touche plus."""
    def policy(obs):
        gap = target - env.n_servers
        if gap >= 5:
            return PLUS5
        if gap > 0:
            return PLUS1
        if gap <= -5:
            return MINUS5
        if gap < 0:
            return MINUS1
        return KEEP
    return policy


def run_episode(env: ClusterEnv, policy) -> dict:
    """Joue un épisode complet et renvoie les métriques (réutilisé pour le DQN à l'étape 5)."""
    obs, _ = env.reset(seed=0)
    total, servers, overloaded, missing, changes = 0.0, [], 0, 0.0, 0
    prev = env.n_servers
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        servers.append(info["n_servers"])
        overloaded += int(info["overloaded"])
        missing += info["missing_servers"]
        changes += info["n_servers"] != prev
        prev = info["n_servers"]
        if terminated or truncated:
            break
    return {
        "recompense": round(total, 1),
        "serveurs_moyens": round(float(np.mean(servers)), 1),
        "saturation_%": round(100 * overloaded / len(servers), 1),
        "serveurs_manquants": round(missing, 1),
        "changements": changes,
    }


def main():
    cpu, mem = load_series("test")
    peak = int(np.ceil(max(cpu.max(), mem.max()) / 100))
    results = {}

    env = ClusterEnv(cpu, mem, random_start=False)
    results["Seuils (HPA)"] = run_episode(env, threshold_policy)

    env = ClusterEnv(cpu, mem, random_start=False)
    results[f"Statique ({peak} serveurs)"] = run_episode(env, make_static_policy(env, peak))

    env = ClusterEnv(cpu, mem, random_start=False)
    env.action_space.seed(0)  # résultat reproductible
    results["Aléatoire"] = run_episode(env, lambda obs: env.action_space.sample())

    print(f"Évaluation sur le jeu de TEST ({len(cpu)} pas de 5 min)\n")
    for name, metrics in results.items():
        print(f"{name:<24}", metrics)


if __name__ == "__main__":
    main()