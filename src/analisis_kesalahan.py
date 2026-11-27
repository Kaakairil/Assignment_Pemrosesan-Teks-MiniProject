"""Analisis kesalahan: kumpulkan FN dan FP, siapkan kolom pola (Tahap 9)."""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import HASIL

logger = logging.getLogger(__name__)

POLA = ["sindiran", "bahasa_campuran", "teks_pendek", "hanya_nama_karakter",
        "menyebut_jalan_cerita_dan_cara", "lainnya"]


def gabung_prediksi(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan prediksi kedua metode berdasarkan id; tandai FN dan FP."""
    gab = pred_a.merge(pred_b, on="id", suffixes=("_a", "_b"), validate="one_to_one")
    gab["fn_a"] = (gab["label_benar_a"] == 1) & (gab["prediksi_a"] == 0)
    gab["fp_a"] = (gab["label_benar_a"] == 0) & (gab["prediksi_a"] == 1)
    gab["fn_b"] = (gab["label_benar_b"] == 1) & (gab["prediksi_b"] == 0)
    gab["fp_b"] = (gab["label_benar_b"] == 0) & (gab["prediksi_b"] == 1)
    return gab[gab[["fn_a", "fp_a", "fn_b", "fp_b"]].any(axis=1)].copy()


def main() -> None:
    """CLI: python -m src.analisis_kesalahan"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ekspor contoh salah klasifikasi kedua metode.")
    parser.add_argument("--pred-a", type=Path, default=HASIL / "prediksi_metode_a.csv")
    parser.add_argument("--pred-b", type=Path, default=HASIL / "prediksi_metode_b_seed42.csv")
    parser.add_argument("--seed-b", type=int, default=42)
    args = parser.parse_args()

    pred_a = pd.read_csv(args.pred_a, encoding="utf-8-sig")
    pred_b = pd.read_csv(
        HASIL / f"prediksi_metode_b_seed{args.seed_b}.csv", encoding="utf-8-sig"
    )
    salah = gabung_prediksi(pred_a, pred_b)

    # Kolom pola dikisi manual oleh pemilik proyek (bagian dari temuan penelitian),
    # diawali kosong; pilihan kategori ada di konstanta POLA.
    salah["pola"] = ""
    salah["jenis_galat"] = ""  # "galat_label" atau "galat_model"; diisi manual bersama pola

    out = HASIL / "analisis_kesalahan.csv"
    salah.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("%d baris salah klasifikasi tersimpan di %s", len(salah), out)

    # Hanya Metode B yang membawa teks_bert; Metode A membawa teks_klasik.
    kolom_teks = {"a": "teks_klasik", "b": "teks_bert"}
    for kolom in ("fn_a", "fp_a", "fn_b", "fp_b"):
        contoh = salah[salah[kolom]][kolom_teks[kolom[-1]]].head(10)
        logger.info("Contoh %s (%d):", kolom, len(salah[salah[kolom]]))
        for t in contoh:
            logger.info("  - %s", t)


if __name__ == "__main__":
    main()
