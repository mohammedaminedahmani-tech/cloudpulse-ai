"""Contrat d'interface partagé entre les 3 pôles de CloudPulse-AI.

Un `MetricEvent` est émis toutes les 5 minutes par la brique Infrastructure (Étudiant 1).
Il est consommé par :
  - l'Étudiant 2 (logs) : jointure avec les logs sur `timestamp` ;
  - l'Étudiant 3 (XAI / agent) : `q_values` pour l'incertitude, `saturated` comme alerte.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1.0"
ACTION_LABELS = ["-5", "-1", "0", "+1", "+5"]  # variation du nombre de serveurs


@dataclass
class MetricEvent:
    timestamp: int                 # secondes depuis le début de la trace Google
    cpu_demand: float              # charge CPU totale (unités : 1 serveur = 100)
    mem_demand: float              # charge mémoire totale (mêmes unités)
    n_servers: int                 # serveurs actifs pendant cette fenêtre
    cpu_util: float                # cpu_demand / capacité (> 1 = saturation)
    mem_util: float
    saturated: bool                # True si la capacité ne suffit pas
    missing_servers: float         # nombre de serveurs qui ont manqué
    action: str                    # décision (prise 5 min avant) qui a fixé n_servers, cf. ACTION_LABELS
    q_values: list[float] = field(default_factory=list)  # valeurs Q de cette décision, même ordre
    source: str = "cloudpulse.infra.autoscaler"
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "MetricEvent":
        return cls(**json.loads(line))