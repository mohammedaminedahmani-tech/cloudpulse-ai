# Contrat d'interface — Pôle Infrastructure (Étudiant 1)

## Ce que produit ce pôle
Un flux d'événements JSON, **1 ligne = 1 fenêtre de 5 minutes** :
`data/stream/metrics.jsonl` (généré par `python -m src.infra.stream`).

Le format est défini dans `src/common/schemas.py` (`MetricEvent`, version 1.0).

## Exemple
```json
{"timestamp": 260700, "cpu_demand": 2590.05, "mem_demand": 2697.1, "n_servers": 35,
 "cpu_util": 0.74, "mem_util": 0.7706, "saturated": false, "missing_servers": 0.0,
 "action": "+5", "q_values": [-13.11, -7.4807, -6.6457, -6.2733, -5.7126],
 "source": "cloudpulse.infra.autoscaler", "schema_version": "1.0"}
```

## Champs
| Champ | Type | Signification |
|---|---|---|
| `timestamp` | int | Secondes depuis le début de la trace Google 2011 |
| `cpu_demand`, `mem_demand` | float | Charge totale (1 serveur = 100 unités) |
| `n_servers` | int | Serveurs actifs pendant la fenêtre |
| `cpu_util`, `mem_util` | float | Utilisation ; > 1 = saturation |
| `saturated` | bool | La capacité ne suffit pas → **alerte** |
| `missing_servers` | float | Nombre de serveurs manquants |
| `action` | str | Décision (prise 5 min avant) parmi `-5, -1, 0, +1, +5` |
| `q_values` | list[5] | Valeurs Q de l'agent, même ordre que les actions |

## Lire le flux en Python
```python
from src.common.schemas import MetricEvent
events = [MetricEvent.from_json(l) for l in open("data/stream/metrics.jsonl", encoding="utf-8")]
alertes = [e for e in events if e.saturated]
```

## Pistes d'intégration
- **Étudiant 2 (logs)** : joindre les logs Loghub aux événements sur `timestamp` (fenêtres de 5 min).
- **Étudiant 3 (XAI / agent)** :
  - confiance de l'agent = écart entre les 2 meilleures `q_values` (faible écart = incertitude) ;
  - `saturated == true` = déclencheur de l'agent de remédiation RAG.

## Règle
Ne pas modifier `MetricEvent` sans incrémenter `SCHEMA_VERSION` et prévenir l'équipe.