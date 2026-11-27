"""Gabung label final dan pembagian data per video (Tahap 5).

Pembagian per video_id, bukan acak: komentar dari video yang sama bisa
bocor antar split kalau dibagi acak. Lihat dokumen proyek (docs/update) bagian 2.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_LABELED, DATA_INTERIM

logger = logging.getLogger(__name__)

# Video IDs dari dokumen proyek v2.1 tabel pembagian (bagian 2)
TRAIN_VIDEO_IDS = {
    "45r3OyaxvFE", "QSPPyA1736U", "B2Bon2X1jPQ", "03qhcshfeMY",
    "lF_QYSm2T-Q", "u_Lb6sh9N0M", "MWQtXTTAgeM", "z6sBvvg_004",
}
VAL_VIDEO_IDS = {"zYTPbx83wCQ", "wJVeWiqMJ-M"}
TEST_VIDEO_IDS = {"5m-Xi8lgvWY"}
ALL_VALID_IDS = TRAIN_VIDEO_IDS | VAL_VIDEO_IDS | TEST_VIDEO_IDS
SPOILER_EXAMPLE_IDS = set(str(i) for i in range(1, 201))

# Data uji tidak boleh dipakai untuk memilih parameter apa pun (grid search,
# threshold, panjang token, dll). Hanya untuk evaluasi akhir sekali.


def validasi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reject baris tidak valid: status != valid, label kosong, spoiler_example, video tidak di tabel."""
    awal = len(df)
    
    # 1. Reject status != "valid"
    if "status" in df.columns:
        invalid_status = df[df["status"] != "valid"]
        if len(invalid_status) > 0:
            logger.warning("Reject %d baris dengan status != 'valid'", len(invalid_status))
            df = df[df["status"] == "valid"]
    
    # 2. Reject label_final kosong atau NaN
    kosong = df["label_final"].isna() | (df["label_final"] == "")
    if kosong.sum() > 0:
        logger.warning("Reject %d baris dengan label_final kosong", kosong.sum())
        df = df[~kosong]
    
    # 3. Reject spoiler_example (id 1-200)
    if df["id"].astype(str).isin(SPOILER_EXAMPLE_IDS).any():
        n_example = df["id"].astype(str).isin(SPOILER_EXAMPLE_IDS).sum()
        logger.warning("Reject %d baris dari spoiler_example", n_example)
        df = df[~df["id"].astype(str).isin(SPOILER_EXAMPLE_IDS)]
    
    # 4. Reject video_id tidak di tabel (atau kolom tidak ada)
    if "video_id" not in df.columns:
        raise ValueError("Kolom 'video_id' tidak ada. File belum siap untuk split per video.")
    
    invalid_video = ~df["video_id"].isin(ALL_VALID_IDS)
    if invalid_video.sum() > 0:
        logger.warning("Reject %d baris dengan video_id tidak di tabel pembagian", invalid_video.sum())
        df = df[~invalid_video]
    
    logger.info("Validasi: %d baris awal → %d baris valid", awal, len(df))
    return df.reset_index(drop=True)


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
    """Validasi, buang ambigu (-1), lalu bagi per video_id. Cek kebocoran ID."""
    df = validasi_data(df)

    n_ambigu = int((df["label_final"] == -1).sum())
    df = df[df["label_final"] != -1].reset_index(drop=True)
    logger.info("Dibuang ambigu (-1): %d; sisa %d baris", n_ambigu, len(df))

    train = df[df["video_id"].isin(TRAIN_VIDEO_IDS)]
    val = df[df["video_id"].isin(VAL_VIDEO_IDS)]
    test = df[df["video_id"].isin(TEST_VIDEO_IDS)]
    assert len(train) + len(val) + len(test) == len(df), "Ada baris tidak masuk split mana pun"

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
