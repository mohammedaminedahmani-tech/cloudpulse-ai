"""Étape 6 : Edge AI - export ONNX du réseau DQN et quantification INT8.

Le réseau de l'agent (6 -> 64 -> 64 -> 5) est reconstruit en ONNX à partir de ses poids,
puis quantifié en INT8. On compare : taille, latence, accord des décisions et
performance sur le jour de test.
"""
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.infra.baselines import run_episode
from src.infra.env import ClusterEnv, load_series

MODEL_PATH = Path("models/dqn_best/best_model.zip")
EDGE_DIR = Path("models/edge")
FP32_PATH = EDGE_DIR / "autoscaler_fp32.onnx"
INT8_PATH = EDGE_DIR / "autoscaler_int8.onnx"
DOCS = Path("docs")


def get_weights():
    """Récupère les poids du réseau Q appris par Stable-Baselines3 : 3 couches (W, b)."""
    from stable_baselines3 import DQN
    model = DQN.load(MODEL_PATH, device="cpu")
    sd = model.q_net.q_net.state_dict()  # Linear(0) ReLU(1) Linear(2) ReLU(3) Linear(4)
    return [(sd[f"{i}.weight"].numpy(), sd[f"{i}.bias"].numpy()) for i in (0, 2, 4)]


def build_onnx(layers, path: Path):
    """Écrit le réseau en ONNX : (MatMul + Add) -> Relu -> (MatMul + Add) -> Relu -> (MatMul + Add)."""
    nodes, inits, x = [], [], "obs"
    for k, (w, b) in enumerate(layers):
        inits += [numpy_helper.from_array(w.T.astype(np.float32), f"W{k}"),
                  numpy_helper.from_array(b.astype(np.float32), f"B{k}")]
        nodes += [helper.make_node("MatMul", [x, f"W{k}"], [f"mm{k}"]),
                  helper.make_node("Add", [f"mm{k}", f"B{k}"], [f"h{k}"])]
        x = f"h{k}"
        if k < len(layers) - 1:
            nodes.append(helper.make_node("Relu", [x], [f"r{k}"]))
            x = f"r{k}"
    nodes.append(helper.make_node("Identity", [x], ["q_values"]))

    graph = helper.make_graph(
        nodes, "dqn_autoscaler",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", 6])],
        [helper.make_tensor_value_info("q_values", TensorProto.FLOAT, ["batch", 5])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # compatible avec onnxruntime
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def onnx_policy(session):
    return lambda obs: int(session.run(None, {"obs": obs[None].astype(np.float32)})[0].argmax())


def collect_observations(env: ClusterEnv, policy) -> np.ndarray:
    """Observations rencontrées sur le jour de test (pour comparer les décisions FP32 / INT8)."""
    obs, _ = env.reset(seed=0)
    all_obs = [obs]
    while True:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        all_obs.append(obs)
        if terminated or truncated:
            return np.array(all_obs, dtype=np.float32)


def latency_us(session, obs: np.ndarray, n: int = 2000) -> float:
    """Temps moyen d'UNE décision (batch de 1), comme sur une sonde de monitoring."""
    one = obs[:1]
    for _ in range(100):  # échauffement
        session.run(None, {"obs": one})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"obs": one})
    return (time.perf_counter() - t0) / n * 1e6


def main(layers=None):
    layers = layers if layers is not None else get_weights()

    # 1) Export FP32 puis quantification INT8 (poids stockés sur 8 bits au lieu de 32)
    build_onnx(layers, FP32_PATH)
    quantize_dynamic(str(FP32_PATH), str(INT8_PATH), weight_type=QuantType.QInt8)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # une sonde edge a peu de coeurs
    sessions = {name: ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                for name, p in [("FP32", FP32_PATH), ("INT8", INT8_PATH)]}

    # 2) Mêmes observations pour les deux modèles
    cpu, mem = load_series("test")
    obs = collect_observations(ClusterEnv(cpu, mem, random_start=False), onnx_policy(sessions["FP32"]))
    actions = {name: s.run(None, {"obs": obs})[0].argmax(axis=1) for name, s in sessions.items()}

    rows = {}
    for name, path in [("FP32", FP32_PATH), ("INT8", INT8_PATH)]:
        metrics = run_episode(ClusterEnv(cpu, mem, random_start=False), onnx_policy(sessions[name]))
        rows[name] = {
            "taille_Ko": round(path.stat().st_size / 1024, 1),
            "latence_us": round(latency_us(sessions[name], obs), 1),
            "accord_decisions_%": round(100 * float(np.mean(actions[name] == actions["FP32"])), 1),
            **metrics,
        }

    table = pd.DataFrame(rows).T
    print(table.to_string())
    DOCS.mkdir(exist_ok=True)
    table.to_csv(DOCS / "resultats_edge.csv")
    print(f"\nModèles sauvegardés dans {EDGE_DIR}")


if __name__ == "__main__":
    main()