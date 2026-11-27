"""Pôle 2 - Baseline : TF-IDF + régression logistique.

Pour chaque tâche : F1 macro sur val et test (3 graines pour la régression).
Test croisé : entraînement sur une tâche, test sur l'autre (écart de domaine).
Produit docs/pole2/resultats_baseline.csv

Usage : python -m src.logs.baseline
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

DATA = Path("data/logs")
OUT = Path("docs/pole2")
TASKS = ["A", "B"]
SEEDS = [0, 1, 2]


def load(task: str, part: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"task_{task}_{part}.csv")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def fit(train: pd.DataFrame, seed: int):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X, train["label"])
    return vec, clf


def score(vec, clf, df: pd.DataFrame) -> dict:
    pred = clf.predict(vec.transform(df["text"]))
    return {
        "f1_macro": f1_score(df["label"], pred, average="macro"),
        "accuracy": accuracy_score(df["label"], pred),
        "rappel_anomalie": recall_score(df["label"], pred, pos_label=1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {t: {p: load(t, p) for p in ["train", "val", "test"]} for t in TASKS}
    rows = []
    for tr in TASKS:
        for te in TASKS:
            for part in (["val", "test"] if tr == te else ["test"]):
                res = [score(*fit(data[tr]["train"], s), data[te][part]) for s in SEEDS]
                row = {"entraine_sur": tr, "teste_sur": te, "partie": part}
                for k in res[0]:
                    vals = [r[k] for r in res]
                    row[k] = round(float(np.mean(vals)), 4)
                    row[k + "_std"] = round(float(np.std(vals)), 4)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "resultats_baseline.csv", index=False)
    print(df[["entraine_sur", "teste_sur", "partie", "f1_macro", "f1_macro_std",
              "rappel_anomalie"]].to_string(index=False))


if __name__ == "__main__":
    main()