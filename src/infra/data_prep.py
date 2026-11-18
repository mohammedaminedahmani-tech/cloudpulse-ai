"""Étape 1 : traces Google 2011 (task_usage) -> série temporelle de charge du cluster."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/task_usage")
OUT_FILE = Path("data/processed/cluster_load.csv")
WINDOW_US = 300 * 1_000_000  # 5 minutes en microsecondes


def load_and_aggregate(path: Path) -> pd.DataFrame:
    """Lit un fichier task_usage et renvoie la charge totale par fenêtre de 5 min."""
    df = pd.read_csv(path, header=None, usecols=[0, 5, 6])
    df.columns = ["start_time", "cpu", "mem"]
    df["window"] = df["start_time"] // WINDOW_US
    return df.groupby("window")[["cpu", "mem"]].sum()


def main():
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .csv.gz dans {RAW_DIR}")

    # 1) Agrégation fichier par fichier (économise la RAM)
    parts = []
    for path in files:
        print("Traitement de", path.name)
        parts.append(load_and_aggregate(path))

    # 2) Ré-agrégation : une fenêtre peut être coupée entre deux fichiers
    charge = pd.concat(parts).groupby(level=0).sum()

    # 3) La dernière fenêtre continue dans un fichier non téléchargé -> on la retire
    charge = charge.iloc[:-1]

    # 4) Temps en secondes (le numéro de fenêtre est dans l'index)
    charge["timestamp"] = charge.index * 300

    # 5) Sauvegarde
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    charge[["timestamp", "cpu", "mem"]].to_csv(OUT_FILE, index=False)

    print(f"{len(charge)} fenêtres sauvegardées dans {OUT_FILE}")
    print(f"Durée couverte : {len(charge) * 5 / 60:.1f} heures")


if __name__ == "__main__":
    main()