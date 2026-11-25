"""Kesepakatan antarlabel: Cohen's kappa pada subset 100 komentar (Tahap 4)."""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import DATA_LABELED, HASIL, MIN_KAPPA, ROOT, SEED

logger = logging.getLogger(__name__)


def muat_lembar(path: Path) -> pd.DataFrame:
    """Muat lembar label dan pastikan kolom minimal ada."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    for kolom in ("id", "label"):
        if kolom not in df.columns:
            raise ValueError(f"Kolom '{kolom}' tidak ada di {path}")
    return df


def subset_kesepakatan(n: int = 100) -> list[str]:
    """Ambil daftar ID subset kesepakatan (disimpan saat pembuatan lembar)."""
    path = DATA_LABELED / "subset_kesepakatan.csv"
    if not path.exists():
        raise FileNotFoundError(f"Subset kesepakatan belum dibuat: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")["id"].astype(str).tolist()


def hitung_kappa(lembar_a: pd.DataFrame, lembar_b: pd.DataFrame, ids: list[str]) -> dict:
    """Hitung Cohen's kappa dengan label -1 dikeluarkan, plus tabel silang."""
    a = lembar_a.set_index(lembar_a["id"].astype(str))["label"]
    b = lembar_b.set_index(lembar_b["id"].astype(str))["label"]
    gab = pd.concat([a, b], axis=1, keys=["a", "b"]).loc[[str(i) for i in ids]]
    pasangan = gab[(gab["a"] != -1) & (gab["b"] != -1) & gab["a"].notna() & gab["b"].notna()]
    if pasangan.empty:
        raise ValueError("Tidak ada pasangan label valid (semua -1 atau kosong).")
    kappa = cohen_kappa_score(pasangan["a"], pasangan["b"])
    crosstab = pd.crosstab(pasangan["a"], pasangan["b"])
    return {"kappa": kappa, "n": len(pasangan), "crosstab": crosstab, "gabungan": gab}


def main() -> None:
    """CLI: python -m src.kesepakatan"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Hitung Cohen's kappa antar dua pelabel.")
    parser.add_argument("--lembar-a", type=Path, default=DATA_LABELED / "lembar_label_a.csv")
    parser.add_argument("--lembar-b", type=Path, default=DATA_LABELED / "lembar_label_b.csv")
    args = parser.parse_args()

    ids = subset_kesepakatan()
    hasil = hitung_kappa(muat_lembar(args.lembar_a), muat_lembar(args.lembar_b), ids)

    HASIL.mkdir(parents=True, exist_ok=True)
    out = HASIL / "kesepakatan.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Kappa Cohen: {hasil['kappa']:.4f}\n")
        f.write(f"Pasangan dipakai: {hasil['n']} (label -1 dikeluarkan)\n")
        f.write(f"Ambang minimum: {MIN_KAPPA}\n\n")
        f.write("Tabel silang:\n")
        f.write(hasil["crosstab"].to_string())
    logger.info("Kappa %.4f (n=%d), tersimpan di %s", hasil["kappa"], hasil["n"], out)
    print(hasil["crosstab"])

    if hasil["kappa"] < MIN_KAPPA:
        raise SystemExit(
            f"Kappa {hasil['kappa']:.4f} < {MIN_KAPPA}. Perbaiki panduan label, "
            "diskusikan perbedaan, lalu ulangi. Jangan lanjut ke pelabelan penuh."
        )
    logger.info("Kappa memenuhi ambang. Pelabelan penuh boleh dilanjutkan.")


if __name__ == "__main__":
    main()
