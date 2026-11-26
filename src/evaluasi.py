"""Evaluasi kedua metode: metrik kelas spoiler, confusion matrix, McNemar (Tahap 8)."""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

from src.config import HASIL

logger = logging.getLogger(__name__)


def metrik_spoiler(y_true, y_pred) -> dict:
    """Precision, recall, F1 kelas 1, macro F1, dan accuracy."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    return {
        "precision_spoiler": p[0],
        "recall_spoiler": r[0],
        "f1_spoiler": f1[0],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": (pd.Series(y_true) == pd.Series(y_pred)).mean(),
    }


def simpan_confusion_matrix(y_true, y_pred, judul: str, path: Path) -> None:
    """Simpan confusion matrix sebagai PNG dengan judul dan label kelas jelas."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bukan spoiler", "Spoiler"],
        yticklabels=["Bukan spoiler", "Spoiler"],
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Benar")
    plt.title(judul)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Confusion matrix tersimpan: %s", path)


def uji_mcnemar(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> dict:
    """McNemar pada prediksi uji yang sama dari kedua metode (gabung per id)."""
    gab = pred_a.merge(pred_b, on="id", suffixes=("_a", "_b"), validate="one_to_one")
    tabel = pd.crosstab(
        (gab["prediksi_a"] == gab["label_benar_a"]),
        (gab["prediksi_b"] == gab["label_benar_b"]),
    ).reindex(index=[True, False], columns=[True, False], fill_value=0)
    hasil = mcnemar(tabel.values, exact=False, correction=True)
    return {"statistik": hasil.statistic, "p_value": hasil.pvalue, "tabel": tabel}


def main() -> None:
    """CLI: python -m src.evaluasi"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluasi kedua metode pada data uji yang sama.")
    parser.add_argument("--pred-a", type=Path, default=HASIL / "prediksi_metode_a.csv")
    parser.add_argument("--pred-b-glob", type=str, default="prediksi_metode_b_seed*.csv")
    args = parser.parse_args()

    HASIL.mkdir(parents=True, exist_ok=True)
    pred_a = pd.read_csv(args.pred_a, encoding="utf-8-sig")

    # Metode B: rata-rata ± std antar seed (skor tiap seed, metrik agregat).
    path_b = sorted(HASIL.glob(args.pred_b_glob))
    if not path_b:
        raise FileNotFoundError(f"Tidak ada prediksi Metode B dengan pola {args.pred_b_glob} di {HASIL}")
    preds_b = [pd.read_csv(p, encoding="utf-8-sig") for p in path_b]

    baris = [{"metode": "A (TF-IDF + LogReg)", **metrik_spoiler(pred_a["label_benar"], pred_a["prediksi"])}]

    # Metrik per seed, lalu dilaporkan sebagai rata-rata ± simpangan baku.
    metrik_seed = [metrik_spoiler(pb["label_benar"], pb["prediksi"]) for pb in preds_b]
    df_seed = pd.DataFrame(metrik_seed)

    # Prediksi konsensus: rata-rata skor antar seed, ambang 0.5.
    # PENTING: petakan lewat kolom id — assignment Series berindeks id ke DataFrame
    # ber-RangeIndex akan menghasilkan NaN semua.
    skor_rata = sum(p.set_index("id")["skor"] for p in preds_b) / len(preds_b)
    gab = preds_b[0][["id", "label_benar"]].copy()
    gab["skor_rata"] = gab["id"].map(skor_rata)
    gab["prediksi"] = (gab["skor_rata"] >= 0.5).astype(int)
    pred_b_mean = gab

    baris.append(
        {
            "metode": "B (IndoBERT, mean±std antar seed)",
            "precision_spoiler": df_seed["precision_spoiler"].mean(),
            "recall_spoiler": df_seed["recall_spoiler"].mean(),
            "f1_spoiler": df_seed["f1_spoiler"].mean(),
            "macro_f1": df_seed["macro_f1"].mean(),
            "accuracy": df_seed["accuracy"].mean(),
            "f1_std_antar_seed": df_seed["f1_spoiler"].std(),
            "recall_std_antar_seed": df_seed["recall_spoiler"].std(),
            "n_seed": len(df_seed),
        }
    )
    baris[0]["f1_std_antar_seed"] = None
    baris[0]["recall_std_antar_seed"] = None
    baris[0]["n_seed"] = 1

    df_metrik = pd.DataFrame(baris)
    df_metrik.to_csv(HASIL / "metrik.csv", index=False, encoding="utf-8-sig")
    logger.info("Metrik tersimpan:\n%s", df_metrik.to_string(index=False))
    logger.info("Metrik konsensus B (mean skor antar seed):\n%s",
                metrik_spoiler(pred_b_mean["label_benar"], pred_b_mean["prediksi"]))

    simpan_confusion_matrix(
        pred_a["label_benar"], pred_a["prediksi"], "Confusion Matrix — Metode A (TF-IDF + LogReg)",
        HASIL / "confusion_matrix_metode_a.png",
    )
    simpan_confusion_matrix(
        pred_b_mean["label_benar"], pred_b_mean["prediksi"], "Confusion Matrix — Metode B (IndoBERT, konsensus seed)",
        HASIL / "confusion_matrix_metode_b.png",
    )

    # McNemar: A vs B konsensus (data uji yang sama).
    m = uji_mcnemar(pred_a, pred_b_mean)
    with open(HASIL / "mcnemar.txt", "w", encoding="utf-8") as f:
        f.write("Uji McNemar (Metode A vs Metode B konsensus)\n")
        f.write(f"Statistik: {m['statistik']:.4f}\n")
        f.write(f"P-value: {m['p_value']:.6f}\n\n")
        f.write("Tabel kontingensi (benar/salah per metode):\n")
        f.write(m["tabel"].to_string())
    logger.info("McNemar: stat=%.4f p=%.6f", m["statistik"], m["p_value"])


if __name__ == "__main__":
    main()
