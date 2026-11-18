"""Étape 4 : entraînement de l'agent DQN d'autoscaling."""
import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.infra.env import ClusterEnv, load_series

MODEL_DIR = Path("models")
VAL_START = 720  # 60 h x 12 pas : les 12 dernières heures du train servent de validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cpu, mem = load_series("train")
    train_env = Monitor(ClusterEnv(cpu[:VAL_START], mem[:VAL_START]))
    val_env = Monitor(ClusterEnv(cpu[VAL_START:], mem[VAL_START:], random_start=False))

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50_000,          # mémoire des expériences passées
        learning_starts=1_000,       # on joue un peu avant d'apprendre
        batch_size=64,
        gamma=0.99,                  # importance du futur (anticipation)
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.3,    # 30 % du temps pour passer de l'exploration à l'exploitation
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[64, 64]),  # petit réseau : 6 entrées -> 64 -> 64 -> 5 actions
        seed=args.seed,
        verbose=0,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=str(MODEL_DIR / "dqn_best"),
        log_path=str(MODEL_DIR / "dqn_logs"),
        eval_freq=5_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(MODEL_DIR / "dqn_final")
    print("Meilleur modèle sauvegardé dans", MODEL_DIR / "dqn_best" / "best_model.zip")


if __name__ == "__main__":
    main()