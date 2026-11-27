# CloudPulse-AI

Système AIOps autonome de supervision, d'autoscaling et de remédiation pour architectures Cloud.
Projet du module **Intelligence Artificielle Avancée**.

## Objectif global

Les infrastructures Cloud subissent des pics de charge imprévisibles, des pannes en cascade et des flux de logs illisibles. CloudPulse-AI regroupe trois briques qui coopèrent :

1. **Décider** combien de serveurs garder allumés (autoscaling par Deep RL) et rendre ce modèle assez léger pour une sonde (Edge AI).
2. **Comprendre les logs** (LoRA/QLoRA) et **continuer d'apprendre** sans oublier (EWC).
3. **Faire confiance et agir** : expliquer les alertes (XAI, incertitude) et laisser un agent DevOps proposer une remédiation (RAG).

## Les 3 pôles

| Pôle | Thèmes | Dossier | Statut |
|---|---|---|---|
| **1. Infrastructure, Autoscaling & Edge** | T7 Deep RL · T12 Edge AI | `src/infra`, `src/edge` | ✅ Terminé |
| **2. Analyse des logs & apprentissage continu** | T2 LoRA/QLoRA · T13 EWC | `src/logs` | ✅ Terminé |
| 3. Confiance, XAI & agent DevOps | T9 XAI · T3 RAG/Agents | `src/agent` | ⏳ À faire |

Les pôles communiquent via des flux d'événements JSON :

```
 Pôle 1                         Pôle 2                           Pôle 3
 metrics.jsonl  ───────────────►  stream_logs.py  ─────────────►  XAI + agent RAG
 (MetricEvent, 1 ligne          incidents.jsonl                   (lit les deux flux,
  = 5 min de charge)            (IncidentEvent, 1 ligne            jointure sur timestamp)
                                 = 1 log classé)
```

Contrats : **[docs/INTERFACE.md](docs/INTERFACE.md)** et `src/common/schemas.py` (`MetricEvent`, pôle 1) ; `src/logs/incident.py` (`IncidentEvent`, pôle 2).

## Sommaire

1. [Installation](#installation)
2. [Pôle 1 — Reproduire les résultats](#pôle-1--reproduire-les-résultats)
3. [Pôle 2 — Logs, LoRA/QLoRA et EWC](#pôle-2--logs-loraqlora-et-apprentissage-continu-ewc)
4. [Guide Google Colab](#guide-google-colab-gpu)
5. [Passation : comment le pôle 3 continue](#passation--comment-le-pôle-3-continue)
6. [Structure du dépôt](#structure-du-dépôt)
7. [Contribuer](#contribuer)

---

## Installation

```bash
git clone https://github.com/mohammedaminedahmani-tech/cloudpulse-ai.git
cd cloudpulse-ai
python -m venv .venv
# Windows (cmd) : .venv\Scripts\activate.bat | PowerShell : .venv\Scripts\Activate.ps1
# Linux/Mac     : source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

Résultat attendu : **10 tests passent** (4 du pôle 1, 3 sur EWC/métriques, 3 sur l'incident/calibration).

> **Windows** : lancez toujours les commandes (`git`, `python -m src...`, `pytest`) depuis la racine `cloudpulse-ai`. L'invite doit se terminer par `cloudpulse-ai>`. Évitez de cloner dans `C:\Windows\System32`.
>
> Le pôle 2 a besoin d'un **GPU** pour l'entraînement : voir le [guide Colab](#guide-google-colab-gpu). La préparation des données, la baseline, les graphiques et les tests tournent sur CPU.

---

## Pôle 1 — Reproduire les résultats

*(Section écrite par l'étudiant 1, conservée telle quelle.)*

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

### Perspectives (pôle 1)
- Réduire les oscillations (83 changements) : augmenter `change_penalty`.
- Donner à l'agent un historique plus long, ou tester PPO.
- **Distillation** : un réseau élève 6 → 16 → 5 imitant le DQN, pour une sonde encore plus légère.
- Quantification statique (avec calibration) pour un vrai gain de latence.

---

## Pôle 2 — Logs, LoRA/QLoRA et apprentissage continu (EWC)

**Auteur : étudiant 2.** Thèmes : **T2** (LLM, PEFT, LoRA, QLoRA) et **T13** (apprentissage continu, oubli catastrophique, EWC).

### Objectif

Un modèle de langage qui **lit des logs système et dit s'ils sont normaux ou anormaux**, puis qui **apprend un nouveau type de logs sans oublier l'ancien**. Il produit enfin un flux d'incidents (`incidents.jsonl`) avec un **score** et une **confiance** que le pôle 3 peut exploiter.

Deux questions scientifiques :
- **T2** : peut-on adapter un LLM avec 0,1 % de ses poids (LoRA) ou en 4 bits (QLoRA) sans perdre en qualité par rapport à un réentraînement complet ?
- **T13** : quand on apprend une 2ᵉ tâche, combien de la 1ʳᵉ est oublié, et EWC le corrige-t-il ?

### Notions utilisées

- **LoRA** : les poids du modèle sont gelés ; on entraîne de petites matrices de rang 8 ajoutées aux couches `q_proj` et `v_proj`.
- **QLoRA** : LoRA sur un modèle de base quantifié en 4 bits (NF4), pour réduire la mémoire GPU.
- **Oubli catastrophique** : un modèle réentraîné sur une nouvelle tâche perd ses performances sur l'ancienne.
- **EWC (Elastic Weight Consolidation)** : après la tâche A, on estime l'importance de chaque poids (matrice de Fisher diagonale) ; pendant la tâche B on pénalise `λ/2 · Σ F·(θ − θ*)²`. λ règle le compromis stabilité/plasticité.
- **Calibration par température** : un paramètre T qui rend la confiance du modèle plus fiable ; mesurée par l'ECE.

### Modèle et scénario

- Modèle de base : **Qwen2.5-0.5B-Instruct** + tête de classification (2 classes), longueur max 256 jetons.
- **Tâche A : HDFS_v1** (logs de système de fichiers distribué). Une entrée = la séquence d'événements d'un bloc.
- **Tâche B : BGL** (supercalculateur Blue Gene/L). Une entrée = un message.
- Étiquettes communes aux deux tâches (indispensable pour EWC) : **0 = normal, 1 = anomalie**.
- Scénario continu : apprendre A, puis B, puis mesurer l'oubli de A.

### Les données (`src/logs/data_prep.py`)

| | Tâche A (HDFS) | Tâche B (BGL) |
|---|---|---|
| Source | `HDFS_v1.zip` (Loghub/Zenodo), lu **sans extraction** | `BGL.log` (Loghub/Zenodo) |
| Étiquettes réelles | `anomaly_label.csv` par bloc | 1ʳᵉ colonne : `-` = normal, sinon alerte |
| Découpage train / val / test | 1 665 / 367 / 380 | 1 126 / 239 / 362 |
| Anomalies en test | **190** | **28** |

Prétraitement : masquage des IP, UUID, nœuds, hexadécimaux, chemins, nombres ; suppression des textes présents avec les deux étiquettes ; **découpage par groupe** (squelette du message) pour que des variantes triviales d'un même message ne se retrouvent pas à la fois en train et en test.

Les CSV produits sont versionnés dans `data/logs/` (`task_A_*.csv`, `task_B_*.csv`, colonnes `text,label`). Les données brutes (Loghub) **ne sont pas dans le dépôt** : elles se placent dans `~/Projets/loghub_full/` si vous voulez régénérer les CSV.

**Jeux écartés (et pourquoi)** : OpenStack (les anomalies sont par instance de VM, pas par ligne : 0 anomalie distincte après nettoyage), Thunderbird (27,7 M de lignes lues, seulement 35 messages d'alerte distincts), et BGL découpé par catégories d'alerte (trop peu de messages distincts par catégorie). Ce sont des propriétés des jeux, pas des bugs.

### Explication de chaque fichier de `src/logs/`

| Fichier | Où il tourne | Rôle |
|---|---|---|
| `data_prep.py` | PC (CPU) | Lit HDFS (zip) et BGL, masque, étiquette, découpe par groupe, écrit `data/logs/task_{A,B}_{train,val,test}.csv`. `python -m src.logs.data_prep [A\|B]` |
| `baseline.py` | PC (CPU) | TF-IDF + régression logistique sur A et B, plus test croisé A→B et B→A. Écrit `docs/pole2/resultats_baseline.csv` |
| `finetune.py` | **Colab (GPU)** | Full fine-tuning, LoRA ou QLoRA sur une tâche ; mesure F1, paramètres entraînables, pic mémoire GPU, temps, taille. Écrit `docs/pole2/resultats_finetuning.csv` |
| `ewc.py` | importé | Fisher diagonale (`compute_fisher`), pénalité EWC (`ewc_penalty`), `snapshot` / `restore` des poids |
| `metrics.py` | importé | Matrice R et métriques continues : `average_final`, `forgetting` (oubli), `bwt` |
| `continual.py` | **Colab (GPU)** | Scénario A → B avec 4 méthodes : naïf, EWC (balayage de λ), replay (10 % de A), joint. Écrit `docs/pole2/resultats_continu.csv` |
| `plots.py` | PC (CPU) | Graphiques du compromis λ, des méthodes, de la matrice R ; résumé moyenne ± écart-type |
| `calibration.py` | importé | `fit_temperature` (température par LBFGS), `ece`, `softmax` |
| `train_final.py` | **Colab (GPU)** | Entraîne le modèle final A → B avec EWC (λ = 10 000), calibre la température, sauvegarde l'adaptateur dans `models/pole2/final_ewc/`, écrit `docs/pole2/calibration.csv` |
| `incident.py` | importé | Contrat de sortie `IncidentEvent` (voir plus bas) |
| `stream_logs.py` | **Colab (GPU)** | Lit `metrics.jsonl`, associe des logs de test à chaque fenêtre, les classe et écrit `data/stream/incidents.jsonl` |

Tests : `tests/test_logs_continual.py` (pénalité EWC nulle à θ*, valeur connue après déplacement, métriques sur une matrice connue) et `tests/test_logs_incident.py` (aller-retour JSON, ECE, température).

### Résultats

#### Baseline TF-IDF + régression logistique (F1 macro)

| Entraîné sur → testé sur | F1 macro | Rappel anomalie |
|---|---|---|
| A → A | 0,953 | 0,995 |
| B → B | 0,894 | 0,893 |
| A → B | 0,072 | 1,000 |
| B → A | 0,339 | 0,005 |

Le **décalage de domaine est énorme** : un modèle entraîné sur HDFS est inutilisable sur BGL et inversement. C'est ce qui justifie le scénario continu.

#### T2 : full FT vs LoRA vs QLoRA (tâche A, 3 graines, moyenne ± écart-type)

| Mode | F1 test | Params entraînables | Pic mémoire GPU | Temps | Taille |
|---|---|---|---|---|---|
| Full FT | **0,980 ± 0,003** | 494 M (100 %) | 9,94 Go | 287 s | ~1 976 Mo (estimation) |
| LoRA | 0,973 ± 0,004 | 542 k (0,11 %) | 4,42 Go | 99 s | **2,2 Mo** |
| QLoRA | 0,971 ± 0,003 | 542 k | **3,41 Go** | 143 s | **2,2 Mo** |

LoRA et QLoRA sont **comparables** au full FT en n'entraînant que 0,11 % des poids ; QLoRA économise environ 23 % de mémoire par rapport à LoRA mais est environ 45 % plus lent. La taille du full FT est une estimation (paramètres × 4 octets), non mesurée sur fichier. Les écarts sont de l'ordre de quelques exemples sur 380 : écrire « comparable », pas « meilleur ».

#### T13 : apprentissage continu A → B (LoRA, 3 graines)

| Méthode | F1 sur A après B | F1 sur B | Oubli de A |
|---|---|---|---|
| Naïf | 0,812 | 0,932 | 0,161 ± 0,131 |
| EWC λ = 10 | 0,888 | 0,925 | 0,085 ± 0,062 |
| EWC λ = 100 | 0,917 | 0,926 | 0,056 ± 0,047 |
| EWC λ = 1 000 | 0,945 | 0,930 | 0,027 ± 0,032 |
| **EWC λ = 10 000** | **0,959** | 0,921 | **0,014 ± 0,006** |
| EWC λ = 100 000 | 0,970 | 0,852 | 0,003 ± 0,005 |
| Replay (10 % de A) | 0,948 | 0,915 | 0,025 ± 0,015 |
| Joint (borne supérieure) | 0,974 | 0,929 | n/a |

Graphiques : `docs/pole2/continu_lambda.png` (compromis λ), `continu_methodes.png` (comparaison), `continu_matrice_R.png` (matrice R). Tableau complet : `docs/pole2/resultats_continu_resume.csv`.

**Lecture :**
- L'**oubli catastrophique est réel** : l'apprentissage naïf de BGL fait perdre en moyenne 16 points de F1 sur HDFS, mais avec une très forte variabilité selon la graine (0,03 à 0,29).
- **EWC réduit l'oubli** d'un facteur ~11 (λ = 10 000) **sans stocker aucune donnée de A**, à un niveau comparable au replay.
- À **λ = 100 000** le modèle est trop rigide : F1 sur B tombe à 0,85. La zone utile est **λ entre 1 000 et 10 000** ; la validation ne désigne pas un λ unique de façon nette.
- Sur la validation, le replay est meilleur qu'EWC : ne pas conclure que « EWC bat le replay ».

#### Calibration (modèle final LoRA + EWC, λ = 10 000, graine 0)

| Test | n | F1 macro | ECE avant | ECE après (T = 1,785) |
|---|---|---|---|---|
| A | 380 | 0,955 | 0,013 | 0,038 |
| B | 362 | 0,933 | 0,019 | 0,015 |
| A+B | 742 | 0,960 | 0,014 | 0,017 |

Le modèle est **déjà bien calibré** (ECE ≈ 0,014) : la température n'apporte pas de gain mesurable. La température est tout de même sauvegardée (`temperature.json`) et utilisée dans le flux.

#### Flux d'incidents (`stream_logs.py`)

Sur 302 fenêtres et 906 logs : **rappel 0,931, précision 0,976, exactitude 0,974**. Part de logs signalés « anomalie » : 0,649 en fenêtres tendues contre 0,045 en fenêtres calmes.

> **Limite importante : le lien charge ↔ logs est simulé.** Les traces Google n'ont pas de logs. Pour chaque fenêtre de 5 min, 3 logs sont tirés des jeux de **test** (jamais vus à l'entraînement) : 70 % d'anomalies si la fenêtre est saturée ou tendue (utilisation ≥ 0,9), 3 % sinon. Le contraste tendues/calmes est donc une conséquence de la simulation, pas une corrélation observée. Le flux démontre l'**intégration technique** et la détection, rien de plus.

### Le contrat `IncidentEvent` (`src/logs/incident.py`)

Une ligne de `data/stream/incidents.jsonl` = un log classé :

```json
{"timestamp": 260700, "log_text": "RAS KERNEL FATAL program interrupt", "label": "anomalie",
 "anomaly_score": 0.9312, "confidence": 0.9312, "model": "lora+ewc",
 "source": "cloudpulse.logs.classifier", "schema_version": "1.0"}
```

| Champ | Type | Signification |
|---|---|---|
| `timestamp` | int | Même repère que `MetricEvent.timestamp` (secondes depuis le début de la trace Google) : **clé de jointure** |
| `log_text` | str | Log nettoyé analysé |
| `label` | str | `"normal"` ou `"anomalie"` |
| `anomaly_score` | float | Probabilité de la classe anomalie (0 à 1) |
| `confidence` | float | Probabilité max, **calibrée par température** (0 à 1) |
| `model` | str | `"lora+ewc"` |
| `source`, `schema_version` | str | `cloudpulse.logs.classifier`, `"1.0"` |

Lecture en Python :

```python
from src.logs.incident import IncidentEvent
incidents = [IncidentEvent.from_json(l) for l in open("data/stream/incidents.jsonl", encoding="utf-8")]
anomalies = [i for i in incidents if i.label == "anomalie"]
```

### Reproduire le pôle 2

| Étape | Où | Commande | Produit |
|---|---|---|---|
| 1. Données (Loghub dans `~/Projets/loghub_full`) | PC | `python -m src.logs.data_prep` | `data/logs/*.csv` (déjà dans le dépôt) |
| 2. Baseline | PC | `python -m src.logs.baseline` | `docs/pole2/resultats_baseline.csv` |
| 3. Fine-tuning | Colab | `python -m src.logs.finetune --mode lora --task A --seeds 0 1 2` (idem `qlora`, `full`) | `docs/pole2/resultats_finetuning.csv` |
| 4. Apprentissage continu | Colab | `python -m src.logs.continual --seeds 0 1 2` | `docs/pole2/resultats_continu.csv` |
| 5. Graphiques | PC | `python -m src.logs.plots` | `docs/pole2/continu_*.png` |
| 6. Modèle final | Colab | `python -m src.logs.train_final --seed 0 --lam 10000` | `models/pole2/final_ewc/`, `docs/pole2/calibration.csv` |
| 7. Flux d'incidents | Colab | `python -m src.logs.stream_logs` | `data/stream/incidents.jsonl` |

Les étapes 1 à 7 ont déjà été faites : leurs résultats sont dans le dépôt. L'adaptateur final (`models/pole2/final_ewc/`, 2,2 Mo) est versionné ; le modèle de base Qwen est téléchargé automatiquement depuis Hugging Face.

### Limites du pôle 2

- Étiquettes issues de Loghub (par bloc pour HDFS, par ligne pour BGL). **BGL a peu de messages réellement distincts** : 28 anomalies en test, donc F1 sur B bruité. L'oubli sur A (190 anomalies en test) est la mesure fiable.
- Seulement **3 graines** : écarts-types indicatifs, aucun test statistique. La graine ne change pas le découpage des données.
- Le naïf est très instable (oubli de 0,03 à 0,29 selon la graine) ; la Fisher varie beaucoup d'une graine à l'autre (max 0,49 contre ~70), donc un même λ n'a pas le même effet partout.
- La taille du full FT est estimée ; le pic mémoire de QLoRA n'inclut pas la quantification initiale.
- Le flux d'incidents repose sur un lien charge/logs **simulé**.

### Perspectives (pôle 2)
- Plus de graines et un test statistique ; augmenter le nombre de logs BGL distincts.
- Tester EWC sur QLoRA, ou sur 3 tâches ou plus.
- Fisher moyennée sur plusieurs échantillons / online EWC.
- Vrais logs corrélés à la charge (injection de pannes sur un vrai cluster).

---

## Guide Google Colab (GPU)

Colab est utilisé pour `finetune.py`, `continual.py`, `train_final.py` et `stream_logs.py`. Les commandes de Colab commencent par `!` ou `%` ; les commandes `git`, `copy` et `python -m ...` du PC se tapent dans le terminal Windows, **jamais dans Colab**.

1. Ouvrir `colab.research.google.com`, nouveau notebook, **Exécution > Modifier le type d'exécution > T4 GPU**.
2. Cellule d'installation :

```python
!nvidia-smi | head -n 12
!git clone -b pole2-lora https://github.com/mohammedaminedahmani-tech/cloudpulse-ai.git
%cd cloudpulse-ai
!pip install -q transformers peft bitsandbytes accelerate scikit-learn pandas
!pip install -q -U torchao
```

3. Lancer une commande du tableau ci-dessus, par exemple : `!python -m src.logs.train_final --seed 0 --lam 10000`.
4. **Télécharger tout de suite** les résultats : la session s'efface après une période d'inactivité.

```python
from google.colab import files
files.download("docs/pole2/resultats_continu.csv")
```

Puis sur le PC : `copy "%USERPROFILE%\Downloads\resultats_continu.csv" docs\pole2\resultats_continu.csv`, puis `git add`, `commit`, `push`.

**Problèmes connus :**

| Symptôme | Solution |
|---|---|
| `ImportError: incompatible version of torchao` | `!pip install -q -U torchao`, puis **Exécution > Redémarrer la session** et `%cd /content/cloudpulse-ai` |
| `git pull` : « untracked files would be overwritten » | `!rm -f docs/pole2/<fichier>.csv` puis `!git pull` |
| `No such file or directory: '/content/cloudpulse-ai'` | La session a été réinitialisée : refaire la cellule d'installation |
| `CUDA out of memory` (full FT) | Ajouter `--bs 4` |
| `GPU requis` | Vérifier le type d'exécution T4 |

---

## Passation : comment le pôle 3 continue

Le pôle 3 (**étudiant 3**) travaille dans `src/agent/` : **T9 (XAI, incertitude, calibration)** et **T3 (RAG, agents)**. Il dispose de **deux flux** :

| Flux | Fichier | Contenu |
|---|---|---|
| Métriques (pôle 1) | `data/stream/metrics.jsonl` | charge, saturation, `action`, `q_values` du DQN |
| Incidents (pôle 2) | `data/stream/incidents.jsonl` | logs classés, `anomaly_score`, `confidence` |

**Jointure** : sur `timestamp`. Chaque fenêtre de 5 min de `metrics.jsonl` correspond à 3 logs dans `incidents.jsonl`.

```python
from collections import defaultdict
from src.common.schemas import MetricEvent
from src.logs.incident import IncidentEvent

metrics = [MetricEvent.from_json(l) for l in open("data/stream/metrics.jsonl", encoding="utf-8") if l.strip()]
by_ts = defaultdict(list)
for l in open("data/stream/incidents.jsonl", encoding="utf-8"):
    if l.strip():
        i = IncidentEvent.from_json(l)
        by_ts[i.timestamp].append(i)

for m in metrics:
    logs = by_ts.get(m.timestamp, [])
    # m.saturated, m.q_values  +  logs[k].label / anomaly_score / confidence
```

**Pistes pour le pôle 3 :**
- **Incertitude / T9** : combiner la confiance de l'agent DQN (écart entre les 2 meilleures `q_values`, déjà indiqué dans `docs/INTERFACE.md`) avec la `confidence` du classifieur de logs. Mesurer l'ECE de l'ensemble.
- **XAI / T9** : expliquer pourquoi un log est classé « anomalie » (attribution par jeton, perturbation du texte) sur le modèle `models/pole2/final_ewc`.
- **Déclencheur de l'agent / T3** : `saturated == true` **et** au moins un log `label == "anomalie"` avec une confiance élevée → l'agent interroge la documentation (RAG) et propose un plan de remédiation.

**Charger le modèle de logs** (adaptateur LoRA + température) :

```python
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

D = Path("models/pole2/final_ewc")
cfg = json.loads((D / "temperature.json").read_text())      # base, température, max_len, lambda
tok = AutoTokenizer.from_pretrained(cfg["base"]); tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForSequenceClassification.from_pretrained(cfg["base"], num_labels=2)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, D).eval()
```

Un exemple complet d'inférence par lots se trouve dans `src/logs/stream_logs.py` (fonction `get_logits`, puis `softmax` de `src/logs/calibration.py` avec la température).

**À savoir :** les logs du flux sont **simulés** (voir limites). Les modèles du pôle 2 sont entraînés sur HDFS et BGL ; ne pas leur demander de généraliser à d'autres formats de logs sans réentraînement (voir la baseline croisée : le décalage de domaine est très fort).

**Règles de collaboration :** créer sa propre branche (par ex. `pole3-agent`), travailler dans `src/agent/`, ne pas modifier `src/common/schemas.py` ni `src/logs/incident.py` sans prévenir l'équipe (incrémenter `SCHEMA_VERSION` en cas de changement).

---

## Structure du dépôt

```
data/processed/   série de charge (5 min)           src/infra/   données, simulateur, DQN, flux (pôle 1)
data/logs/        CSV HDFS (A) et BGL (B)           src/edge/    ONNX + quantification (pôle 1)
data/stream/      metrics.jsonl (pôle 1)            src/common/  contrat MetricEvent partagé
                  incidents.jsonl (pôle 2)          src/logs/    pôle 2 : données, LoRA, EWC, calibration, flux
models/           DQN, ONNX (pôle 1)                src/agent/   → pôle 3
                  pole2/final_ewc (adaptateur LoRA)  tests/       10 tests
docs/             résultats et graphiques pôle 1    notebooks/   exploration des données
docs/pole2/       résultats et graphiques pôle 2
```

## Contribuer

1. Créez une branche : `git checkout -b pole3-agent` (le pôle 2 a utilisé `pole2-lora`).
2. Ne modifiez pas `src/common/schemas.py` ni `src/logs/incident.py` sans prévenir l'équipe (voir `docs/INTERFACE.md`).
3. Vérifiez que `python -m pytest tests -q` passe (10 tests) avant chaque push.
4. Poussez votre branche puis ouvrez une Pull Request vers `main`.
