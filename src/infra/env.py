"""Étape 2 : simulateur de cluster pour l'autoscaling (environnement Gymnasium)."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

DATA_FILE = Path("data/processed/cluster_load.csv")


def load_series(split: str = "train", train_hours: float = 72, path: Path = DATA_FILE):
    """Découpage TEMPOREL : les `train_hours` premières heures pour l'entraînement, le reste pour le test."""
    df = pd.read_csv(path)
    cut = df["timestamp"].iloc[0] + train_hours * 3600
    part = df[df["timestamp"] < cut] if split == "train" else df[df["timestamp"] >= cut]
    return part["cpu"].to_numpy(np.float32), part["mem"].to_numpy(np.float32)


class ClusterEnv(gym.Env):
    """Toutes les 5 min, l'agent choisit d'ajouter ou de retirer des serveurs.

    Un changement ne prend effet qu'au pas suivant (temps de démarrage) :
    l'agent doit donc anticiper la charge.
    """

    ACTIONS = np.array([-5, -1, 0, 1, 5])  # variation du nombre de serveurs

    def __init__(self, cpu, mem, episode_len=288, server_capacity=100.0,
                 min_servers=10, max_servers=60,
                 overload_penalty=10.0, change_penalty=0.5, random_start=True):
        super().__init__()
        self.cpu = np.asarray(cpu, dtype=np.float32)
        self.mem = np.asarray(mem, dtype=np.float32)
        self.capacity = server_capacity
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.overload_penalty = overload_penalty  # 1 serveur manquant coûte 10x un serveur allumé
        self.change_penalty = change_penalty      # évite les oscillations
        self.random_start = random_start
        # Épisode complet si on ne tire pas de départ au hasard (mode évaluation)
        self.episode_len = episode_len if random_start else len(self.cpu) - 3
        self.load_scale = server_capacity * max_servers  # pour ramener les charges vers [0, 1]

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # [util_cpu, util_mem, n_serveurs/max, cpu(t), cpu(t-1), cpu(t-2)]
        self.observation_space = spaces.Box(low=0.0, high=3.0, shape=(6,), dtype=np.float32)

    def _obs(self):
        t, cap = self.t, self.n_servers * self.capacity
        return np.clip(np.array([
            self.cpu[t] / cap,
            self.mem[t] / cap,
            self.n_servers / self.max_servers,
            self.cpu[t] / self.load_scale,
            self.cpu[t - 1] / self.load_scale,
            self.cpu[t - 2] / self.load_scale,
        ], dtype=np.float32), 0.0, 3.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            self.t = int(self.np_random.integers(2, len(self.cpu) - self.episode_len - 1))
        else:
            self.t = 2
        self.steps = 0
        # On démarre avec juste assez de serveurs pour la charge actuelle
        need = max(self.cpu[self.t], self.mem[self.t]) / self.capacity
        self.n_servers = int(np.clip(np.ceil(need), self.min_servers, self.max_servers))
        return self._obs(), {}

    def step(self, action):
        old = self.n_servers
        self.n_servers = int(np.clip(old + self.ACTIONS[action], self.min_servers, self.max_servers))
        change = abs(self.n_servers - old)

        # On avance de 5 min : la nouvelle capacité affronte la NOUVELLE charge
        self.t += 1
        self.steps += 1
        cap = self.n_servers * self.capacity
        missing = (max(0.0, self.cpu[self.t] - cap) + max(0.0, self.mem[self.t] - cap)) / self.capacity

        cost = self.n_servers + self.overload_penalty * missing + self.change_penalty * change
        reward = -cost / self.max_servers

        truncated = self.steps >= self.episode_len
        info = {
            "n_servers": self.n_servers,
            "cpu_demand": float(self.cpu[self.t]),
            "mem_demand": float(self.mem[self.t]),
            "overloaded": missing > 0,
            "missing_servers": float(missing),
        }
        return self._obs(), float(reward), False, truncated, info