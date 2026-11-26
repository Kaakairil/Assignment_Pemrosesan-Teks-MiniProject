"""Metode A: TF-IDF + Logistic Regression (Tahap 6).

Hyperparameter dipilih dengan melatih pada train.csv dan mengevaluasi pada
val.csv (tanpa StratifiedKFold acak, agar komentar video yang sama tidak bocor
antar lipatan). Setelah terkunci, model dilatih ulang pada train.csv dan
dievaluasi SEKALI pada test.csv.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.config import HASIL, MODELS, SEED

logger = logging.getLogger(__name__)

GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [None, 10000],
    "clf__C": [0.1, 1, 10],
}


def buat_pipeline(params: dict | None = None) -> Pipeline:
    """Pipeline TF-IDF + Logistic Regression sesuai spesifikasi."""
    default = {
        "tfidf__ngram_range": (1, 2),
        "tfidf__max_features": None,
        "clf__C": 1.0,
    }
    default.update(params or {})
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=2,
                    sublinear_tf=True,
                    ngram_range=default["tfidf__ngram_range"],
                    max_features=default["tfidf__max_features"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                    C=default["clf__C"],
                ),
            ),
        ]
    )


def buat_grid(params_grid: dict | None = None) -> list[dict]:
    """Semua kombinasi hyperparameter dari GRID."""
    g = params_grid or GRID
    return [
        {
            "tfidf__ngram_range": ng,
            "tfidf__max_features": mf,
            "clf__C": c,
        }
        for ng in g["tfidf__ngram_range"]
        for mf in g["tfidf__max_features"]
        for c in g["clf__C"]
    ]


def cari_parameter(
    train: pd.DataFrame, val: pd.DataFrame, params_grid: dict | None = None
) -> tuple[dict, float, list[dict]]:
    """Pilih hyperparameter terbaik: fit di train, skor F1 kelas 1 di val.

    Kembalikan (parameter terbaik, F1 val terbaik, log semua kombinasi).
    """
    log: list[dict] = []
    terbaik: dict | None = None
    f1_terbaik = -1.0
    for params in buat_grid(params_grid):
        pipe = buat_pipeline(params)
        pipe.fit(train["teks_klasik"], train["label_final"])
        pred = pipe.predict(val["teks_klasik"])
        f1 = f1_score(val["label_final"], pred, pos_label=1)
        log.append({**params, "ngram": str(params["tfidf__ngram_range"]), "f1_val": round(float(f1), 4)})
        logger.info("F1 val=%.4f  %s", f1, params)
        if f1 > f1_terbaik:
            f1_terbaik, terbaik = f1, params
    assert terbaik is not None
    logger.info("Parameter terbaik: %s (F1 val=%.4f)", terbaik, f1_terbaik)
    return terbaik, float(f1_terbaik), log


def prediksi_uji(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    """Prediksi data uji; keluarkan kolom video_id/id/label/prediksi/skor."""
    pred = model.predict(test["teks_klasik"])
    skor = model.predict_proba(test["teks_klasik"])[:, 1]
    return pd.DataFrame(
        {
            "video_id": test["video_id"],
            "id": test["id"],
            "teks_klasik": test["teks_klasik"],
            "label_benar": test["label_final"],
            "prediksi": pred,
            "skor": skor,
        }
    )


def main() -> None:
    """CLI: python -m src.metode_a"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Latih Metode A (TF-IDF + LogReg).")
    parser.add_argument("--train", type=Path, default=None, help="CSV train (default data/labeled/train.csv)")
    parser.add_argument("--val", type=Path, default=None, help="CSV val (default data/labeled/val.csv)")
    parser.add_argument("--test", type=Path, default=None, help="CSV test (default data/labeled/test.csv)")
    args = parser.parse_args()

    from src.config import DATA_LABELED

    train = pd.read_csv(args.train or DATA_LABELED / "train.csv", encoding="utf-8-sig")
    val = pd.read_csv(args.val or DATA_LABELED / "val.csv", encoding="utf-8-sig")
    test = pd.read_csv(args.test or DATA_LABELED / "test.csv", encoding="utf-8-sig")
    # Kolom kosong (stopword-only) terbaca sebagai NaN; TF-IDF butuh string.
    for df in (train, val, test):
        df["teks_klasik"] = df["teks_klasik"].fillna("")

    # 1) Pilih hyperparameter pada val (uji tidak disentuh sama sekali).
    terbaik, f1_val, log = cari_parameter(train, val)

    # 2) Kunci parameter, latih ulang pada train.csv saja.
    model = buat_pipeline(terbaik)
    model.fit(train["teks_klasik"], train["label_final"])

    # 3) Evaluasi SEKALI pada test.csv.
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "metode_a.joblib")

    HASIL.mkdir(parents=True, exist_ok=True)
    pred = prediksi_uji(model, test)
    pred.to_csv(HASIL / "prediksi_metode_a.csv", index=False, encoding="utf-8-sig")

    f1_uji = f1_score(test["label_final"], pred["prediksi"], pos_label=1)
    with open(HASIL / "params_metode_a.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in terbaik.items()},
                "f1_val": f1_val,
                "f1_test": round(float(f1_uji), 4),
                "log_semua_kombinasi": log,
            },
            f,
            indent=2,
        )

    logger.info("Metode A selesai. F1 uji (sekali) = %.4f", f1_uji)


if __name__ == "__main__":
    main()
