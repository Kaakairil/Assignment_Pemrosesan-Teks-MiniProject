"""Evaluasi kedua metode: metrik kelas spoiler, bootstrap CI, McNemar (Tahap 8).

Metode B dilaporkan sebagai rata-rata ± simpangan baku antar tiga seed.
Tanpa "konsensus seed": tidak ada prediksi B yang dibuat dari gabungan
seed, dan tidak ada seed yang dipilih berdasarkan data uji.
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

from src.config import ALPHA_CI, HASIL, N_BOOTSTRAP, SEED

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


def bootstrap_ci_f1(y_true, y_pred, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """CI 95% untuk F1 kelas 1 dengan bootstrap resample baris test (berbasis id)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    skor = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:  # tanpa spoiler di resample, F1 tak terdefinisi
            continue
        skor.append(f1_score(y_true[idx], y_pred[idx], pos_label=1, zero_division=0))
    bawah, atas = np.percentile(skor, [100 * ALPHA_CI / 2, 100 * (1 - ALPHA_CI / 2)])
    return float(bawah), float(atas)


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

    # Metode B: metrik per seed, dilaporkan sebagai rata-rata ± simpangan baku.
    path_b = sorted(HASIL.glob(args.pred_b_glob))
    if not path_b:
        raise FileNotFoundError(f"Tidak ada prediksi Metode B dengan pola {args.pred_b_glob} di {HASIL}")
    preds_b = [pd.read_csv(p, encoding="utf-8-sig") for p in path_b]
    logger.info("Prediksi B dari %d seed: %s", len(preds_b), [p.name for p in path_b])

    metrik_seed = [metrik_spoiler(pb["label_benar"], pb["prediksi"]) for pb in preds_b]
    df_seed = pd.DataFrame(metrik_seed)
    df_seed["seed"] = [p.stem.replace("prediksi_metode_b_seed", "") for p in path_b]
    df_seed.to_csv(HASIL / "metrik_metode_b_seeds.csv", index=False, encoding="utf-8-sig")

    baris = [
        {
            "metode": "A (TF-IDF + LogReg)",
            **metrik_spoiler(pred_a["label_benar"], pred_a["prediksi"]),
            "n_seed": 1,
        },
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
        },
    ]
    baris[0]["f1_std_antar_seed"] = None
    baris[0]["recall_std_antar_seed"] = None

    # CI 95% bootstrap untuk F1 kelas 1 pada test.csv (dihitung per metode).
    ci_records = []
    ci_a = bootstrap_ci_f1(pred_a["label_benar"], pred_a["prediksi"])
    ci_records.append({"metode": "A", "f1_spoiler": baris[0]["f1_spoiler"], "ci_bawah": ci_a[0], "ci_atas": ci_a[1]})
    # CI bootstrap untuk B: resample per seed, lalu CI dari F1 rata-rata antar seed.
    rng = np.random.default_rng(SEED)
    n = len(pred_a)
    f1_mean_boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        f1s = [
            f1_score(pb["label_benar"].values[idx], pb["prediksi"].values[idx], pos_label=1, zero_division=0)
            for pb in preds_b
        ]
        f1_mean_boot.append(np.mean(f1s))
    ci_b = tuple(np.percentile(f1_mean_boot, [100 * ALPHA_CI / 2, 100 * (1 - ALPHA_CI / 2)]))
    ci_records.append({"metode": "B", "f1_spoiler": baris[1]["f1_spoiler"], "ci_bawah": ci_b[0], "ci_atas": ci_b[1]})
    pd.DataFrame(ci_records).to_csv(HASIL / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    df_metrik = pd.DataFrame(baris)
    df_metrik.to_csv(HASIL / "metrik.csv", index=False, encoding="utf-8-sig")
    logger.info("Metrik tersimpan:\n%s", df_metrik.to_string(index=False))
    logger.info("CI 95%% F1 spoiler — A: [%.4f, %.4f], B: [%.4f, %.4f]", *ci_a, *ci_b)

    # Confusion matrix per metode: A langsung; B per seed (tanpa konsensus seed).
    simpan_confusion_matrix(
        pred_a["label_benar"], pred_a["prediksi"], "Confusion Matrix — Metode A (TF-IDF + LogReg)",
        HASIL / "confusion_matrix_metode_a.png",
    )
    for pb, path in zip(preds_b, path_b):
        seed = path.stem.replace("prediksi_metode_b_seed", "")
        simpan_confusion_matrix(
            pb["label_benar"], pb["prediksi"], f"Confusion Matrix — Metode B (IndoBERT, seed {seed})",
            HASIL / f"confusion_matrix_metode_b_seed{seed}.png",
        )

    # McNemar: A vs B per seed (data uji yang sama, prediksi B asli per seed).
    with open(HASIL / "mcnemar.txt", "w", encoding="utf-8") as f:
        f.write("Uji McNemar (Metode A vs Metode B, prediksi uji yang sama)\n")
        f.write("McNemar dijalankan per seed untuk B; tanpa konsensus seed.\n\n")
        for pb, path in zip(preds_b, path_b):
            seed = path.stem.replace("prediksi_metode_b_seed", "")
            m = uji_mcnemar(pred_a, pb)
            f.write(f"--- Seed {seed} ---\n")
            f.write(f"Statistik: {m['statistik']:.4f}\n")
            f.write(f"P-value: {m['p_value']:.6f}\n")
            f.write("Tabel kontingensi (benar/salah per metode):\n")
            f.write(m["tabel"].to_string() + "\n\n")
            logger.info("McNemar seed %s: stat=%.4f p=%.6f", seed, m["statistik"], m["p_value"])


if __name__ == "__main__":
    main()
