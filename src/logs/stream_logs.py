"""Pôle 2 - Flux d'incidents : data/stream/metrics.jsonl -> data/stream/incidents.jsonl.

Pour chaque fenêtre de 5 min, on associe 3 logs synthétiques tirés des jeux de TEST
(HDFS + BGL). Le lien métriques/logs est SIMULÉ (les traces Google n'ont pas de logs) :
fenêtre saturée ou tendue -> 70 % de logs anormaux, sinon 3 %.
Chaque log est classé par le modèle LoRA+EWC, avec une confiance calibrée.

Usage :
    python -m src.logs.stream_logs                 # GPU
    python -m src.logs.stream_logs --device cpu --limit 100
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.schemas import MetricEvent
from src.logs.calibration import softmax
from src.logs.finetune import BASE
from src.logs.incident import LABELS, IncidentEvent

MODEL_DIR = Path("models/pole2/final_ewc")
METRICS = Path("data/stream/metrics.jsonl")
OUT = Path("data/stream/incidents.jsonl")
DATA = Path("data/logs")


def load_model(device):
    cfg = json.loads((MODEL_DIR / "temperature.json").read_text())
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, MODEL_DIR).to(device).eval()
    return tok, model, cfg


@torch.no_grad()
def get_logits(model, tok, texts, device, max_len):
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--logs_per_window", type=int, default=3)
    ap.add_argument("--tension", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="nombre de fenêtres (0 = toutes)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok, model, cfg = load_model(a.device)
    T, max_len = cfg["temperature"], cfg["max_len"]
    print(f"Modèle {cfg['methode']} (λ={cfg['lambda']:g}), température T={T:.3f}, "
          f"appareil : {a.device}")

    pool = pd.concat([pd.read_csv(DATA / f"task_{t}_test.csv") for t in "AB"])
    pool["text"] = pool["text"].fillna("").astype(str)
    normal = pool[pool["label"] == 0]["text"].tolist()
    anom = pool[pool["label"] == 1]["text"].tolist()

    events = [MetricEvent.from_json(l) for l in open(METRICS, encoding="utf-8") if l.strip()]
    if a.limit:
        events = events[:a.limit]
    rng = random.Random(a.seed)

    items = []   # (timestamp, texte, vérité, fenêtre_tendue)
    for e in events:
        tendue = e.saturated or max(e.cpu_util, e.mem_util) >= a.tension
        p_anom = 0.70 if tendue else 0.03
        for _ in range(a.logs_per_window):
            is_a = rng.random() < p_anom
            items.append((e.timestamp, rng.choice(anom if is_a else normal), int(is_a), tendue))
    print(f"{len(events)} fenêtres, {len(items)} logs à classer")

    logits = get_logits(model, tok, [i[1] for i in items], a.device, max_len)
    probs = softmax(logits, T)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    truth = np.array([i[2] for i in items])
    pred = probs.argmax(1)
    with open(OUT, "w", encoding="utf-8") as f:
        for (ts, text, _, _), p in zip(items, probs):
            ev = IncidentEvent(timestamp=int(ts), log_text=text,
                               label=LABELS[int(p.argmax())],
                               anomaly_score=round(float(p[1]), 4),
                               confidence=round(float(p.max()), 4),
                               model=cfg["methode"])
            f.write(ev.to_json() + "\n")

    tendue = np.array([i[3] for i in items])
    tp = int(((pred == 1) & (truth == 1)).sum())
    print(f"\nÉcrit : {OUT} ({len(items)} événements)")
    print(f"logs anormaux injectés : {int(truth.sum())} | détectés : {tp} "
          f"(rappel {tp / max(truth.sum(), 1):.3f})")
    print(f"précision : {tp / max((pred == 1).sum(), 1):.3f} | "
          f"exactitude : {(pred == truth).mean():.3f}")
    print(f"part de logs signalés anomalie : fenêtres tendues {pred[tendue].mean():.3f} "
          f"| fenêtres calmes {pred[~tendue].mean():.3f}")


if __name__ == "__main__":
    main()