# CloudPulse-AI

Système AIOps autonome de supervision, d'autoscaling et de remédiation pour architectures Cloud.
Projet du module **Intelligence Artificielle Avancée**.

## Les 3 pôles

| Pôle | Thèmes | Dossier | Statut |
|---|---|---|---|
| **1. Infrastructure, Autoscaling & Edge** | T7 Deep RL · T12 Edge AI | `src/infra`, `src/edge` | ✅ Terminé |
| 2. Analyse des logs & apprentissage continu | T2 LoRA/QLoRA · T13 EWC | `src/logs` | ⏳ À faire |
| 3. Confiance, XAI & agent DevOps | T9 XAI · T3 RAG/Agents | `src/agent` | ⏳ À faire |

Les pôles communiquent via un flux d'événements JSON : voir **[docs/INTERFACE.md](docs/INTERFACE.md)**.

## Installation

```bash
git clone https://github.com/mohammedaminedahmani-tech/cloudpulse-ai.git
cd cloudpulse-ai
python -m venv .venv
# Windows : .venv\Scripts\Activate.ps1   |   Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

> **Sur Google Colab** (utile pour le pôle 2, qui a besoin d'un GPU) :
> `!git clone <url>` puis `%cd cloudpulse-ai` puis `!pip install -r requirements.txt`.

## Pôle 1 — Reproduire les résultats

Toutes les commandes se lancent depuis la racine du projet.

| Étape | Commande | Produit |
|---|---|---|
| 1. Données (≈ 6 Go téléchargés en flux, jamais stockés) | `python -m src.infra.download_traces --end 70` puis `python -m src.infra.data_prep` | `data/processed/cluster_load.csv` |
| 2. Simulateur | `python -m pytest tests -q` | — |
| 3. Baselines | `python -m src.infra.baselines` | — |
| 4. Entraînement DQN (≈ 6 min CPU) | `python -m src.infra.train_dqn --timesteps 200000` | `models/dqn_best/best_model.zip` |
| 5. Évaluation | `python -m src.infra.evaluate` | `docs/resultats_test.csv`, `docs/dqn_vs_hpa.png` |
| 6. Edge (ONNX + INT8) | `python -m src.edge.quantize` | `models/edge/*.onnx`, `docs/resultats_edge.csv` |
| 7. Flux pour les pôles 2 et 3 | `python -m src.infra.stream` | `data/stream/metrics.jsonl` |

Les étapes 1, 4 et 6 sont déjà faites : leurs résultats sont dans le dépôt.

### Données
Traces **Google Cluster 2011** (table `task_usage`, fichiers 0 à 69) agrégées en fenêtres de 5 min :
**1 169 fenêtres, 97,4 h, aucune fenêtre manquante.** Découpage temporel :
60 h d'entraînement, 12 h de validation, ~25 h de test (jamais vues à l'entraînement).

### Simulateur (`src/infra/env.py`)
- 1 pas = 5 min · 1 serveur = 100 unités CPU/mémoire · 10 à 60 serveurs
- 5 actions : −5, −1, 0, +1, +5 serveurs · un changement prend effet au pas suivant (démarrage)
- Récompense : − (serveurs + 50 × serveurs manquants + 20 × [saturation] + 0,5 × changements)

### Résultats sur le jour de test

| Politique | Serveurs moyens | Saturation | Changements |
|---|---|---|---|
| Seuils (type Kubernetes HPA) | 43,3 | 0 % | 15 |
| Statique (dimensionnée au pic) | 42,0 | 0 % | 4 |
| DQN v1 (pénalité ×10) | 32,4 | 20,2 % | 47 |
| **DQN v2 (pénalité ×50 + SLA)** | **34,5** | **3,0 %** | 83 |

**Conclusion :** le DQN v2 utilise **20 % de serveurs en moins** que le HPA, avec une saturation rare.
Le passage de v1 à v2 montre l'effet direct de la conception de la récompense sur le compromis coût/fiabilité.

### Edge AI (`src/edge/quantize.py`)

| Modèle | Taille | Latence (1 décision) | Accord avec FP32 | Saturation |
|---|---|---|---|---|
| ONNX FP32 | 19,6 Ko | 9,4 µs | 100 % | 3,0 % |
| **ONNX INT8** | **7,2 Ko (−63 %)** | 12,6 µs | 94,4 % | 2,6 % |

La quantification INT8 réduit la taille de 63 % **sans dégrader l'autoscaling**.
Elle n'accélère pas ce réseau minuscule (≈ 5 000 paramètres) : la conversion dynamique
des entrées coûte plus que le calcul. Le gain se situe sur la mémoire de la sonde.

## Structure

```
data/processed/   série de charge (5 min)      src/infra/   données, simulateur, DQN, flux
data/stream/      flux JSON pour pôles 2 et 3  src/edge/    ONNX + quantification
models/           DQN + modèles ONNX           src/common/  contrat d'interface partagé
docs/             résultats et graphiques      src/logs/    → pôle 2
notebooks/        exploration des données      src/agent/   → pôle 3
```

## Perspectives (pôle 1)
- Réduire les oscillations (83 changements) : augmenter `change_penalty`.
- Donner à l'agent un historique plus long, ou tester PPO.
- **Distillation** : un réseau élève 6 → 16 → 5 imitant le DQN, pour une sonde encore plus légère.
- Quantification statique (avec calibration) pour un vrai gain de latence.

## Contribuer
1. Créez une branche : `git checkout -b pole2-lora`
2. Ne modifiez pas `src/common/schemas.py` sans prévenir l'équipe (voir `docs/INTERFACE.md`).
3. Vérifiez que `python -m pytest tests -q` passe avant chaque push.