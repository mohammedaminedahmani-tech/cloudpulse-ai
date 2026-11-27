"""Pôle 2 - Modèle final : LoRA, tâche A puis B avec EWC, + calibration par température.

Sauvegarde l'adaptateur dans models/pole2/final_ewc/ (avec temperature.json) et écrit
docs/pole2/calibration.csv (ECE avant/après, sur le test).

Usage (GPU requis) :
    python -m src.logs.train_final --seed 0 --lam 10000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from src.logs.calibration import ece, fit_temperature, softmax
from src.logs.continual import fresh_model, train
from src.logs.ewc import compute_fisher, snapshot
from src.logs.finetune import BASE, load_split

OUT_MODEL = Path("models/pole2/final_ewc")
OUT_DOCS = Path("docs/pole2")


@torch.no_grad()
def get_logits(model, tok, texts, max_len):
    model.eval()
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=10000.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_fisher", type=int, default=200)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU requis : Colab > Exécution > Modifier le type d'exécution > T4.")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    data = {t: {p: load_split(t, p) for p in ("train", "val", "test")} for t in "AB"}

    model = fresh_model(tok, a.seed)
    XA, yA = data["A"]["train"]
    XB, yB = data["B"]["train"]
    train(model, tok, XA, yA, a.epochs, a.bs, a.max_len,
          np.random.RandomState(a.seed), tag="A")
    theta_A = snapshot(model)
    fisher = compute_fisher(model, tok, XA, n_samples=a.n_fisher,
                            max_len=a.max_len, seed=a.seed)
    train(model, tok, XB, yB, a.epochs, a.bs, a.max_len,
          np.random.RandomState(a.seed + 100),
          ewc_state={"fisher": fisher, "theta": theta_A}, lam=a.lam, tag=f"B+EWC lam={a.lam:g}")

    # --- calibration : la température est apprise sur la VALIDATION (A+B) ---
    val_logits, val_y = [], []
    for t in "AB":
        X, y = data[t]["val"]
        val_logits.append(get_logits(model, tok, X, a.max_len))
        val_y.append(y)
    T = fit_temperature(np.concatenate(val_logits), np.concatenate(val_y))
    print(f"\nTempérature apprise sur la validation : T = {T:.3f}")

    # --- évaluation sur le TEST ---
    rows, all_logits, all_y = [], [], []
    for t in "AB":
        X, y = data[t]["test"]
        lg = get_logits(model, tok, X, a.max_len)
        all_logits.append(lg)
        all_y.append(y)
    groups = {"A": (all_logits[0], all_y[0]), "B": (all_logits[1], all_y[1]),
              "A+B": (np.concatenate(all_logits), np.concatenate(all_y))}
    for name, (lg, y) in groups.items():
        p0, p1 = softmax(lg, 1.0), softmax(lg, T)
        rows.append({"ensemble_test": name, "n": len(y), "temperature": round(T, 3),
                     "f1_macro": round(f1_score(y, p0.argmax(1), average="macro"), 4),
                     "ece_avant": round(ece(p0, y), 4), "ece_apres": round(ece(p1, y), 4),
                     "confiance_moy_avant": round(float(p0.max(1).mean()), 4),
                     "confiance_moy_apres": round(float(p1.max(1).mean()), 4)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DOCS / "calibration.csv", index=False)
    model.save_pretrained(OUT_MODEL)
    (OUT_MODEL / "temperature.json").write_text(json.dumps(
        {"temperature": T, "base": BASE, "lambda": a.lam, "seed": a.seed,
         "max_len": a.max_len, "methode": "lora+ewc"}, indent=2))
    print(f"\nAdaptateur et température sauvegardés dans {OUT_MODEL}")


if __name__ == "__main__":
    main()