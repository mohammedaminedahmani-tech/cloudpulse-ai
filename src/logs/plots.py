"""Pôle 2 - Graphiques de l'apprentissage continu (sans GPU).

Lit docs/pole2/resultats_continu.csv et produit dans docs/pole2/ :
  continu_lambda.png  : compromis stabilité/plasticité selon lambda (moyenne ± écart-type)
  continu_methodes.png: comparaison naïf / EWC / replay / joint
  continu_matrice_R.png : matrice R (F1 après chaque tâche) par méthode
  resultats_continu_resume.csv : tableau moyenne ± écart-type

Usage : python -m src.logs.plots
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/pole2")
KEY = ["methode", "lambda", "graine"]


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "resultats_continu.csv")
    return df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["f1_A_apres_B", "f1_B_apres_B", "oubli_A", "f1_val_moyen_final"]
    g = df.groupby(["methode", "lambda"])[cols].agg(["mean", "std"]).round(4)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_graines"] = df.groupby(["methode", "lambda"]).size()
    return g.reset_index()


def plot_lambda(df):
    ewc = df[df["methode"] == "ewc"]
    naif = df[df["methode"] == "naif"]
    lams = sorted(ewc["lambda"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, col, title in [(ax[0], "oubli_A", "Oubli sur A (plus bas = mieux)"),
                          (ax[1], "f1_B_apres_B", "F1 sur B, nouvelle tâche (plus haut = mieux)")]:
        m = [ewc[ewc["lambda"] == l][col].mean() for l in lams]
        s = [ewc[ewc["lambda"] == l][col].std() for l in lams]
        a.errorbar(lams, m, yerr=s, marker="o", capsize=4, label="EWC")
        a.axhline(naif[col].mean(), color="red", ls="--", label="naïf")
        a.set_xscale("log")
        a.set_xlabel("λ")
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Compromis stabilité / plasticité (moyenne ± écart-type, 3 graines)")
    fig.tight_layout()
    fig.savefig(OUT / "continu_lambda.png", dpi=150)
    plt.close(fig)


def plot_methodes(df, lam_choisi=10000.0):
    def pick(method, lam=None):
        d = df[df["methode"] == method]
        return d if lam is None else d[d["lambda"] == lam]
    groups = [("naïf", pick("naif")), (f"EWC λ={lam_choisi:g}", pick("ewc", lam_choisi)),
              ("replay 10 %", pick("replay")), ("joint", pick("joint"))]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (col, lab) in enumerate([("f1_A_apres_B", "F1 sur A (après B)"),
                                    ("f1_B_apres_B", "F1 sur B")]):
        ax.bar(x + (k - 0.5) * w, [g[col].mean() for _, g in groups], w,
               yerr=[g[col].std() for _, g in groups], capsize=4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in groups])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("F1 macro (test)")
    ax.set_title("Méthodes d'apprentissage continu (3 graines)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "continu_methodes.png", dpi=150)
    plt.close(fig)


def plot_matrice(df, lam_choisi=10000.0):
    configs = [("naïf", df[df["methode"] == "naif"]),
               (f"EWC λ={lam_choisi:g}", df[(df["methode"] == "ewc") & (df["lambda"] == lam_choisi)])]
    fig, axes = plt.subplots(1, len(configs), figsize=(8, 3.6))
    for ax, (name, d) in zip(axes, configs):
        R = np.array([[d["f1_A_apres_A"].mean(), d["f1_B_apres_A"].mean()],
                      [d["f1_A_apres_B"].mean(), d["f1_B_apres_B"].mean()]])
        ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                        color="white" if R[i, j] < 0.6 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["test A", "test B"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["après A", "après B"])
        ax.set_title(name)
    fig.suptitle("Matrice R : F1 après chaque tâche (moyenne 3 graines)")
    fig.tight_layout()
    fig.savefig(OUT / "continu_matrice_R.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    print(f"{len(df)} lignes ({df['graine'].nunique()} graines)")
    s = summary(df)
    s.to_csv(OUT / "resultats_continu_resume.csv", index=False)
    print(s.to_string(index=False))
    plot_lambda(df)
    plot_methodes(df)
    plot_matrice(df)
    print("Graphiques écrits dans docs/pole2/")


if __name__ == "__main__":
    main()