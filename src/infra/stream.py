"""Étape 7 : rejoue le jour de test avec l'agent INT8 et produit le flux d'événements JSON.

Sortie : data/stream/metrics.jsonl (1 événement MetricEvent par ligne, 1 ligne = 5 min).
C'est le point d'entrée des Étudiants 2 et 3.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.common.schemas import ACTION_LABELS, MetricEvent
from src.infra.env import DATA_FILE, ClusterEnv, load_series

MODEL_PATH = Path("models/edge/autoscaler_int8.onnx")
OUT_FILE = Path("data/stream/metrics.jsonl")


def main():
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    cpu, mem = load_series("test")
    env = ClusterEnv(cpu, mem, random_start=False)
    obs, _ = env.reset(seed=0)
    # Horodatage réel : le test commence 72 h après la première fenêtre de la trace
    test_start = int(pd.read_csv(DATA_FILE)["timestamp"].iloc[0]) + 72 * 3600

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    n_events, n_saturated = 0, 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        while True:
            q = session.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
            action = int(q.argmax())
            obs, _, terminated, truncated, info = env.step(action)

            cap = info["n_servers"] * env.capacity
            event = MetricEvent(
                timestamp=test_start + 300 * env.t,
                cpu_demand=round(info["cpu_demand"], 2),
                mem_demand=round(info["mem_demand"], 2),
                n_servers=info["n_servers"],
                cpu_util=round(info["cpu_demand"] / cap, 4),
                mem_util=round(info["mem_demand"] / cap, 4),
                saturated=bool(info["overloaded"]),
                missing_servers=round(info["missing_servers"], 2),
                action=ACTION_LABELS[action],
                q_values=[round(float(v), 4) for v in q],
            )
            f.write(event.to_json() + "\n")
            n_events += 1
            n_saturated += event.saturated
            if terminated or truncated:
                break

    print(f"{n_events} événements écrits dans {OUT_FILE} ({n_saturated} saturations)")


if __name__ == "__main__":
    main()