"""Contrat de sortie du pôle 2 (logs) vers le pôle 3.

Un IncidentEvent est émis pour chaque log analysé. Il est volontairement séparé
de src/common/schemas.py (MetricEvent) : à proposer à l'équipe quand il est stable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.0"
LABELS = ["normal", "anomalie"]


@dataclass
class IncidentEvent:
    timestamp: int            # même repère que MetricEvent.timestamp (secondes)
    log_text: str             # log nettoyé analysé
    label: str                # "normal" ou "anomalie"
    anomaly_score: float      # probabilité de la classe anomalie (0..1)
    confidence: float         # probabilité max, idéalement calibrée (0..1)
    model: str = "baseline"   # ex. "tfidf", "lora", "qlora", "lora+ewc"
    source: str = "cloudpulse.logs.classifier"
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "IncidentEvent":
        return cls(**json.loads(line))