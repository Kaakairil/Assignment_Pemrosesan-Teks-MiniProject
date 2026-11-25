"""Buat lembar label A/B dan subset kesepakatan dari komentar_bersih.csv (Tahap 3)."""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_LABELED, DATA_INTERIM, SEED

logger = logging.getLogger(__name__)


def buat_lembar(df: pd.DataFrame, n_kesepakatan: int = 100) -> None:
    """Buat lembar_label_a.csv, lembar_label_b.csv, dan subset_kesepakatan.csv."""
    DATA_LABELED.mkdir(parents=True, exist_ok=True)
    dasar = df[["id", "teks"]].copy()
    dasar["label"] = ""
    dasar["catatan"] = ""

    kolom = ["id", "teks", "label", "catatan"]
    dasar.to_csv(DATA_LABELED / "lembar_label_a.csv", index=False, encoding="utf-8-sig",
                 columns=kolom)
    dasar.to_csv(DATA_LABELED / "lembar_label_b.csv", index=False, encoding="utf-8-sig",
                 columns=kolom)

    subset = df.sample(n=min(n_kesepakatan, len(df)), random_state=SEED)["id"]
    subset.to_frame().to_csv(DATA_LABELED / "subset_kesepakatan.csv", index=False, encoding="utf-8-sig")
    logger.info("Lembar A/B (%d baris) dan subset kesepakatan (%d) dibuat.", len(df), len(subset))


def main() -> None:
    """CLI: python -m src.buat_lembar_label"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Buat lembar pelabelan A/B dan subset kesepakatan.")
    parser.add_argument("--input", type=Path, default=DATA_INTERIM / "komentar_bersih.csv")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    buat_lembar(df, args.n)


if __name__ == "__main__":
    main()
