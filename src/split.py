"""Gabung label final dan pembagian data stratified (Tahap 5)."""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_LABELED, DATA_INTERIM, SEED, TEST_SIZE, VAL_SIZE

logger = logging.getLogger(__name__)

# Data uji tidak boleh dipakai untuk memilih parameter apa pun (grid search,
# threshold, panjang token, dll). Hanya untuk evaluasi akhir sekali.


def gabung_label(path_interim: Path, path_label: Path) -> pd.DataFrame:
    """Gabungkan label_final ke komentar_bersih menjadi komentar_berlabel.csv."""
    bersih = pd.read_csv(path_interim, encoding="utf-8-sig")
    label = pd.read_csv(path_label, encoding="utf-8-sig")
    if "label_final" not in label.columns:
        raise ValueError("Kolom 'label_final' tidak ada di file label.")
    gab = bersih.merge(
        label[["id", "label_final", "catatan"]], on="id", how="left", validate="one_to_one"
    )
    return gab


def buat_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Buang ambigu (-1), lalu stratified split test dan val. Cek kebocoran ID."""
    n_ambigu = int((df["label_final"] == -1).sum())
    df = df[df["label_final"] != -1].reset_index(drop=True)
    logger.info("Dibuang ambigu (-1): %d; sisa %d baris", n_ambigu, len(df))

    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label_final"], random_state=SEED
    )
    train, val = train_test_split(
        train, test_size=VAL_SIZE, stratify=train["label_final"], random_state=SEED
    )

    # Tidak ada ID yang muncul di dua split berbeda.
    id_train, id_val, id_test = set(train["id"]), set(val["id"]), set(test["id"])
    assert not (id_train & id_val), "Kebocoran: ID ada di train dan val"
    assert not (id_train & id_test), "Kebocoran: ID ada di train dan test"
    assert not (id_val & id_test), "Kebocoran: ID ada di val dan test"

    for nama, part in (("train", train), ("val", val), ("test", test)):
        logger.info("%s: %d baris, proporsi label:\n%s", nama, len(part), part["label_final"].value_counts(normalize=True))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    """CLI: python -m src.split"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Gabung label dan buat split stratified.")
    parser.add_argument("--berlabel", type=Path, default=DATA_LABELED / "komentar_berlabel.csv")
    args = parser.parse_args()

    gab = gabung_label(DATA_INTERIM / "komentar_bersih.csv", args.berlabel)
    out = gab.copy()
    out.to_csv(args.berlabel, index=False, encoding="utf-8-sig")

    train, val, test = buat_split(out)
    for nama, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(DATA_LABELED / f"{nama}.csv", index=False, encoding="utf-8-sig")
    logger.info("Split tersimpan di %s", DATA_LABELED)


if __name__ == "__main__":
    main()
