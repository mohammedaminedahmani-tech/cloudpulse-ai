"""Étape 1 bis : télécharger et agréger les traces Google en flux (1 fichier à la fois)."""
import argparse
import urllib.request
from pathlib import Path

from src.infra.data_prep import load_and_aggregate

BASE_URL = "https://commondatastorage.googleapis.com/clusterdata-2011-2/task_usage"
TMP_DIR = Path("data/raw/task_usage")    # fichier brut temporaire
INTERIM_DIR = Path("data/interim")       # petits fichiers agrégés (17 lignes chacun)


def process_part(i: int) -> None:
    name = f"part-{i:05d}-of-00500"
    out = INTERIM_DIR / f"{name}.csv"
    if out.exists():
        print(f"[{name}] déjà traité, on passe")
        return

    gz = TMP_DIR / f"{name}.csv.gz"
    tmp = gz.with_suffix(".download")
    print(f"[{name}] téléchargement...")
    urllib.request.urlretrieve(f"{BASE_URL}/{name}.csv.gz", tmp)
    tmp.rename(gz)  # renommé seulement si le téléchargement est complet

    print(f"[{name}] agrégation...")
    load_and_aggregate(gz).to_csv(out)

    gz.unlink()  # on libère l'espace disque
    print(f"[{name}] OK, fichier brut supprimé")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=70)
    args = parser.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(args.start, args.end):
        process_part(i)


if __name__ == "__main__":
    main()