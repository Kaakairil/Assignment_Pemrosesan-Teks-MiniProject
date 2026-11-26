"""Metode B: IndoBERT fine-tuning dengan Trainer (Tahap 7)."""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import datasets
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import (
    DATA_LABELED,
    HASIL,
    MAX_LEN_BERT,
    MODELS,
    MODEL_BERT,
    SEED,
    SEEDS_BERT,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Tetapkan seed untuk random, numpy, dan torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hitung_bobot_kelas(train: pd.DataFrame) -> torch.Tensor:
    """Bobot kelas dari data latih saja (bukan data uji)."""
    n = len(train)
    bobot = []
    for kelas in (0, 1):
        nk = int((train["label_final"] == kelas).sum())
        bobot.append(n / (2.0 * nk) if nk > 0 else 1.0)
    return torch.tensor(bobot, dtype=torch.float)


class TrainerBobot(Trainer):
    """Trainer dengan class weight pada cross-entropy."""

    def __init__(self, *args, bobot_kelas: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bobot_kelas = bobot_kelas

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=self.bobot_kelas.to(logits.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss


def hitung_f1(pred: "transformers.trainer_utils.EvalPrediction") -> dict:
    """Metrik evaluasi: F1 kelas spoiler (pos_label=1) dan akurasi."""
    from sklearn.metrics import accuracy_score, f1_score

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "f1": f1_score(labels, preds, pos_label=1),
        "accuracy": accuracy_score(labels, preds),
    }


def tokenisasi(teks: list[str], tokenizer) -> dict:
    """Tokenisasi teks_bert dengan truncation dan padding max_length."""
    return tokenizer(teks, truncation=True, max_length=MAX_LEN_BERT, padding="max_length")


def buat_dataset(teks: list[str], label: list[int], tokenizer) -> "datasets.Dataset":
    """Buat HuggingFace Dataset berkolom input_ids/attention_mask/labels."""
    enc = tokenisasi(teks, tokenizer)
    return datasets.Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": [int(l) for l in label],  # Long diperlukan CrossEntropyLoss; float bikin RuntimeError
        }
    )


def latih_satu_seed(seed: int, train: pd.DataFrame, val: pd.DataFrame, tokenizer, lanjut: bool = False) -> AutoModelForSequenceClassification:
    """Fine-tune satu seed; simpan checkpoint terbaik ke models/bert_seed<N>."""
    set_seed(seed)
    out_dir = MODELS / f"bert_seed{seed}"

    ds_train = buat_dataset(train["teks_bert"].tolist(), train["label_final"].tolist(), tokenizer)
    ds_val = buat_dataset(val["teks_bert"].tolist(), val["label_final"].tolist(), tokenizer)

    # Resume dari checkpoint terakhir bila diminta dan ada.
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if out_dir.exists() else []
    titik_mulai = ckpts[-1] if (lanjut and ckpts) else None
    model = AutoModelForSequenceClassification.from_pretrained(
        titik_mulai if titik_mulai else MODEL_BERT, num_labels=2
    )
    bobot = hitung_bobot_kelas(train)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=4,      # kecil: RTX 3050 4GB
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,      # batch efektif tetap 8
        num_train_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,                 # hanya checkpoint terbaik, hemat disk
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=0.01,
        logging_steps=50,
        seed=seed,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = TrainerBobot(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=hitung_f1,
        bobot_kelas=bobot,
    )
    trainer.train(resume_from_checkpoint=str(titik_mulai) if titik_mulai else None)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Seed %d selesai. Checkpoint terbaik di %s", seed, out_dir)
    return model


def prediksi_uji(model, tokenizer, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Prediksi data uji; keluarkan kolom id/label/prediksi/skor."""
    ds = buat_dataset(test["teks_bert"].tolist(), [0] * len(test), tokenizer)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(MODELS / "tmp_pred"), report_to="none", per_device_eval_batch_size=32))
    logits = trainer.predict(ds.remove_columns("labels")).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return pd.DataFrame(
        {
            "id": test["id"],
            "teks_bert": test["teks_bert"],
            "label_benar": test["label_final"],
            "prediksi": logits.argmax(-1),
            "skor": probs,
        }
    )


def main() -> None:
    """CLI: python -m src.metode_b"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT (Metode B).")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--pakai-cpu", action="store_true",
                        help="Paksa jalankan di CPU (lambat). Melewati gate GPU atas izin pemilik proyek.")
    parser.add_argument("--seeds", type=str, default="",
                        help="Seed yang dijalankan, dipisah koma (mis. '44'). Default: semua SEEDS_BERT.")
    parser.add_argument("--lanjut-dari", action="store_true",
                        help="Lanjutkan pelatihan dari checkpoint terakhir di models/bert_seed<N>/ bila ada.")
    args = parser.parse_args()

    from src.config import DATA_LABELED

    if not torch.cuda.is_available():
        if not args.pakai_cpu:
            raise SystemExit(
                "GPU tidak tersedia. Fine-tuning IndoBERT butuh GPU (misalnya di Google Colab). "
                "Gunakan --pakai-cpu untuk menjalankan di CPU atas keputusan sendiri (lambat)."
            )
        logger.warning("Menjalankan di CPU atas permintaan eksplisit — pelatihan akan lambat.")
    else:
        logger.info("GPU terdeteksi: %s", torch.cuda.get_device_name(0))

    train = pd.read_csv(args.train or DATA_LABELED / "train.csv", encoding="utf-8-sig")
    val = pd.read_csv(args.val or DATA_LABELED / "val.csv", encoding="utf-8-sig")
    test = pd.read_csv(args.test or DATA_LABELED / "test.csv", encoding="utf-8-sig")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BERT)

    # Kolom kosong terbaca NaN dari CSV; tokenizer butuh string.
    for part in (train, val, test):
        part["teks_bert"] = part["teks_bert"].fillna("").astype(str)

    # Laporkan proporsi token terpotong; jika >5% tanyakan sebelum melanjutkan.
    panjang = [len(tokenizer(t)["input_ids"]) for t in train["teks_bert"]]
    terpotong = sum(1 for p in panjang if p > MAX_LEN_BERT) / len(panjang)
    logger.info("Proporsi token terpotong: %.2f%%", terpotong * 100)
    if terpotong > 0.05:
        logger.warning("Lebih dari 5%% terpotong — pertimbangkan MAX_LEN_BERT lebih besar.")

    f1_per_seed = []
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else SEEDS_BERT
    for seed in seeds:
        model = latih_satu_seed(seed, train, val, tokenizer, lanjut=args.lanjut_dari)
        pred = prediksi_uji(model, tokenizer, test, seed)
        path = HASIL / f"prediksi_metode_b_seed{seed}.csv"
        pred.to_csv(path, index=False, encoding="utf-8-sig")
        from sklearn.metrics import f1_score

        f1 = f1_score(pred["label_benar"], pred["prediksi"], pos_label=1)
        f1_per_seed.append({"seed": seed, "f1_spoiler": f1})
        logger.info("Seed %d: F1 spoiler = %.4f", seed, f1)
        del model
        torch.cuda.empty_cache()

    # Agregasi dari SEMUA prediksi per-seed yang ada (termasuk run sebelumnya),
    # bukan hanya seed pada run ini — supaya menjalankan subset seed tidak menghapus.
    from sklearn.metrics import f1_score as _f1

    semua = []
    for p in sorted(HASIL.glob("prediksi_metode_b_seed*.csv")):
        dfp = pd.read_csv(p, encoding="utf-8-sig")
        semua.append({"seed": p.stem.replace("prediksi_metode_b_seed", ""), "f1_spoiler": _f1(dfp["label_benar"], dfp["prediksi"], pos_label=1)})
    df_f1 = pd.DataFrame(semua)
    ringkas = pd.DataFrame(
        [{"f1_mean": df_f1["f1_spoiler"].mean(), "f1_std": df_f1["f1_spoiler"].std()}]
    )
    df_f1.to_csv(HASIL / "metrik_metode_b_seeds.csv", index=False)
    ringkas.to_csv(HASIL / "metrik_metode_b_ringkasan.csv", index=False)
    logger.info(
        "Metode B selesai untuk semua seed. F1 mean=%.4f std=%.4f",
        ringkas["f1_mean"].iloc[0],
        ringkas["f1_std"].iloc[0],
    )


if __name__ == "__main__":
    main()
