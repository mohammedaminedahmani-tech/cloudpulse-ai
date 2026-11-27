"""Pôle 2 - Préparation des données : HDFS (tâche A) puis BGL (tâche B).

Classification binaire : 0 = normal, 1 = anomalie.
  Tâche A : HDFS_v1, une entrée = un bloc (séquence de ses événements, lue dans le zip)
  Tâche B : BGL, une entrée = un message d'alerte ou normal
Découpage 70/15/15 PAR GROUPE (squelette) pour éviter les fuites entre train et test.
Produit data/logs/task_{A,B}_{train,val,test}.csv (colonnes text,label).

Usage :
    python -m src.logs.data_prep        # les deux tâches
    python -m src.logs.data_prep A      # HDFS seulement
    python -m src.logs.data_prep B      # BGL seulement
"""
import csv
import io
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FULL = Path.home() / "Projets" / "loghub_full"
HDFS_ZIP = FULL / "HDFS_v1.zip"
BGL_PATH = FULL / "BGL.log"
OUT = Path("data/logs")
SEED = 42
MAX_PER_CLASS = 3000
KEEP_NORMAL_BGL = 0.05
CLASSES = {0: "normal", 1: "anomalie"}

csv.field_size_limit(10**8)

MASKS = [
    (re.compile(r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\bR\d{2}-M\d-N\d+(?:-C:J\d+-U\d+)?\b"), "<NODE>"),
    (re.compile(r"\bbglio\d+\b", re.I), "<IONODE>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<PATH>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
]


def clean(text: str) -> str:
    for rx, tok in MASKS:
        text = rx.sub(tok, text)
    return re.sub(r"\s+", " ", text).strip()[:300]


def skeleton(text: str) -> str:
    t = re.sub(r"<[A-Z]+>", " ", text)
    t = re.sub(r"\b\w*\d\w*\b", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def zip_member(member: str):
    """Lit un fichier du zip en flux (sans extraction sur le disque)."""
    proc = subprocess.Popen(["tar", "-xOf", str(HDFS_ZIP), member],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc, io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="ignore", newline="")


def find_col(header, *keys):
    for i, h in enumerate(header):
        if any(k in h.lower() for k in keys):
            return i
    sys.exit(f"Colonne introuvable ({keys}) dans l'en-tête : {header}")


# --------------------------------------------------------------------- HDFS
def load_hdfs():
    if not HDFS_ZIP.exists():
        sys.exit(f"Fichier introuvable : {HDFS_ZIP}")

    # 1. gabarits d'événements : E1 -> texte court
    proc, f = zip_member("preprocessed/HDFS.log_templates.csv")
    rd = csv.reader(f)
    header = next(rd)
    print("templates, colonnes :", header)
    i_id, i_tp = find_col(header, "eventid", "id"), find_col(header, "template")
    templates = {}
    for row in rd:
        if len(row) > max(i_id, i_tp):
            t = row[i_tp].replace("<*>", " ").replace("[*]", " ").replace("BLOCK* ", "")
            templates[row[i_id].strip()] = re.sub(r"\s+", " ", t).strip()[:60]
    proc.kill()
    print(f"{len(templates)} gabarits d'événements")

    # 2. un bloc = une séquence d'événements
    proc, f = zip_member("preprocessed/Event_traces.csv")
    rd = csv.reader(f)
    header = next(rd)
    print("traces, colonnes :", header)
    i_lab, i_feat = find_col(header, "label"), find_col(header, "feature")
    labels_seen = Counter()
    rows, unknown = [], Counter()
    for row in rd:
        if len(row) <= max(i_lab, i_feat):
            continue
        lab = row[i_lab].strip().lower()
        labels_seen[lab] += 1
        if lab in ("fail", "anomaly", "anomalie", "1"):
            y = 1
        elif lab in ("success", "normal", "0"):
            y = 0
        else:
            continue
        events = [e.strip().strip("'\"") for e in row[i_feat].strip("[] ").split(",") if e.strip()]
        if not events:
            continue
        order, counts = [], Counter()
        for e in events:
            if e not in counts:
                order.append(e)
            counts[e] += 1
        for e in order:
            if e not in templates:
                unknown[e] += 1
        text = " | ".join(templates.get(e, e) + (f" x{counts[e]}" if counts[e] > 1 else "")
                          for e in order)
        rows.append((text, y, ">".join(order)))
    proc.kill()
    print("étiquettes lues :", dict(labels_seen))
    if not rows:
        sys.exit("Aucun bloc lu : étiquettes non reconnues, envoie-moi la ligne ci-dessus.")
    if unknown:
        print("événements sans gabarit :", dict(unknown))
    df = pd.DataFrame(rows, columns=["text", "label", "skel"])
    print(f"blocs lus : {len(df):,} (anomalies : {int(df['label'].sum()):,})")
    return df


# ---------------------------------------------------------------------- BGL
def load_bgl(rng: random.Random):
    if not BGL_PATH.exists():
        sys.exit(f"Fichier introuvable : {BGL_PATH}")
    rows = []
    with open(BGL_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            is_normal = line.startswith("- ")
            if is_normal and rng.random() > KEEP_NORMAL_BGL:
                continue
            parts = line.split(" ", 9)
            if len(parts) < 10:
                continue
            rows.append((clean(" ".join(parts[6:10])), 0 if is_normal else 1))
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["skel"] = df["text"].map(skeleton)
    return df


# ------------------------------------------------------------------ commun
def finalize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    g = df.groupby("text")["label"].agg(["min", "max"])
    conflits = g[g["min"] != g["max"]].index
    print(f"\n=== {name} ===")
    print(f"textes présents avec les 2 étiquettes (écartés) : {len(conflits)}")
    df = df[~df["text"].isin(conflits)].drop_duplicates(subset="text").reset_index(drop=True)
    parts = [d.sample(min(len(d), MAX_PER_CLASS), random_state=SEED)
             for _, d in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def split(df: pd.DataFrame):
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, rest_idx = next(g1.split(df, groups=df["skel"]))
    rest = df.iloc[rest_idx].reset_index(drop=True)
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    v, t = next(g2.split(rest, groups=rest["skel"]))
    return df.iloc[tr], rest.iloc[v], rest.iloc[t]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    tasks = {"A": ("HDFS", lambda: load_hdfs()), "B": ("BGL", lambda: load_bgl(rng))}
    wanted = [a.upper() for a in sys.argv[1:]] or list(tasks)
    for t in wanted:
        if t not in tasks:
            sys.exit(f"Tâche inconnue : {t} (choix : A, B)")
        name, loader = tasks[t]
        print(f"\nLecture de {name}...")
        df = finalize(loader(), f"Tâche {t} ({name})")
        types = df[df["label"] == 1]["skel"].nunique()
        print("après nettoyage :", df["label"].map(CLASSES).value_counts().to_dict(),
              f"| types d'anomalie distincts : {types}")
        train, val, test = split(df)
        for part_name, part in [("train", train), ("val", val), ("test", test)]:
            part[["text", "label"]].to_csv(OUT / f"task_{t}_{part_name}.csv", index=False)
            c = part["label"].value_counts().to_dict()
            print(f"   {part_name:5s}: {len(part):5d} lignes | normal {c.get(0, 0)} | "
                  f"anomalie {c.get(1, 0)} | types d'anomalie "
                  f"{part[part['label'] == 1]['skel'].nunique()}")
        print("exemples d'anomalies :")
        for text in df[df["label"] == 1]["text"].head(2):
            print("   ", text[:200])


if __name__ == "__main__":
    main()