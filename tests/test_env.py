from gymnasium.utils.env_checker import check_env

from src.infra.env import ClusterEnv, load_series


def test_decoupage_temporel():
    cpu_train, _ = load_series("train")
    cpu_test, _ = load_series("test")
    assert len(cpu_train) == 864          # 72 h x 12 fenêtres de 5 min
    assert len(cpu_test) > 0


def test_env_conforme_gymnasium():
    env = ClusterEnv(*load_series("train"))
    check_env(env)                        # vérificateur officiel de Gymnasium


def test_episode_agent_aleatoire():
    env = ClusterEnv(*load_series("train"))
    env.reset(seed=0)
    pas, total = 0, 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        pas += 1
        assert env.min_servers <= info["n_servers"] <= env.max_servers
        if terminated or truncated:
            break
    assert pas == 288                     # 1 épisode = 1 journée
    print(f"\nAgent aléatoire : récompense totale = {total:.1f}, serveurs finaux = {info['n_servers']}")