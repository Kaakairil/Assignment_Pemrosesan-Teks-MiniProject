"""Metode A: TF-IDF + Logistic Regression dengan GridSearchCV (Tahap 6)."""
import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import HASIL, MODELS, SEED

logger = logging.getLogger(__name__)


def buat_pipeline() -> Pipeline:
    """Pipeline TF-IDF + Logistic Regression sesuai spesifikasi."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]
    )


def cari_parameter(train: pd.DataFrame) -> GridSearchCV:
    """GridSearchCV pada data latih saja, scoring F1 kelas spoiler."""
    grid = GridSearchCV(
        buat_pipeline(),
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [None, 10000],
            "clf__C": [0.1, 1, 10],
        },
        scoring=make_scorer(f1_score, pos_label=1),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    grid.fit(train["teks_klasik"], train["label_final"])
    logger.info("Parameter terbaik: %s (cv F1=%.4f)", grid.best_params_, grid.best_score_)
    return grid


def prediksi_uji(grid: GridSearchCV, test: pd.DataFrame) -> pd.DataFrame:
    """Prediksi data uji dengan model terbaik; keluarkan kolom id/label/prediksi/skor."""
    pred = grid.best_estimator_.predict(test["teks_klasik"])
    skor = grid.best_estimator_.predict_proba(test["teks_klasik"])[:, 1]
    return pd.DataFrame(
        {
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
    parser.add_argument("--test", type=Path, default=None, help="CSV test (default data/labeled/test.csv)")
    args = parser.parse_args()

    from src.config import DATA_LABELED

    train = pd.read_csv(args.train or DATA_LABELED / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(args.test or DATA_LABELED / "test.csv", encoding="utf-8-sig")
    # Kolom kosong (stopword-only) terbaca sebagai NaN; TF-IDF butuh string.
    train["teks_klasik"] = train["teks_klasik"].fillna("")
    test["teks_klasik"] = test["teks_klasik"].fillna("")

    grid = cari_parameter(train)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, MODELS / "metode_a.joblib")

    HASIL.mkdir(parents=True, exist_ok=True)
    pred = prediksi_uji(grid, test)
    pred.to_csv(HASIL / "prediksi_metode_a.csv", index=False, encoding="utf-8-sig")

    with open(HASIL / "params_metode_a.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": grid.best_params_, "cv_f1": grid.best_score_}, f, indent=2)

    logger.info("Metode A selesai. Model dan prediksi tersimpan.")


if __name__ == "__main__":
    main()
