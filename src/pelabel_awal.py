"""Pelabel awal BERBAS MODEL/HEURISTIK — BUKAN label manusia (Tahap 3-5, sementara).

CLAUDE.md: "Jangan membuat label otomatis dan menyebutnya sebagai label manusia.
Jika saya meminta bantuan pelabelan awal dengan model, tandai kolom sebagai
label_model dan jelaskan di laporan."

Label yang dihasilkan skrip ini disimpan di kolom `label_model`. Hasil Metode A/B
yang memakai label ini adalah PRELIMINAR, bukan temuan final. Cohen's kappa tidak
dihitung (tak bermakna antara dua salinan heuristik yang sama) — dicatat sebagai
dilewati, bukan dipalsukan.
"""
import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.config import DATA_INTERIM, DATA_LABELED

logger = logging.getLogger(__name__)

# Heuristik berbasis pola nyata di data: komentar yang MENJELASKAN peristiwa
# cerita (nasib karakter, isi adegan, isi ending) -> kandidat spoiler.
# Opini/reaksi/sapaan tanpa isi cerita -> 0. Pendek -> ambigu.
_POLA_SPOILER = [
    # nasib karakter bernama
    r"\b(donie|donnie|doni|lucia|jason|fatimah|mandy|mendez)\w* (tewas|mati|dibunuh|ditembak|selamat|fitnah|ditangkap|dipenjara|pengkhianat|khianat)\b",
    r"\b(tewas|mati|dibunuh|ditembak|selamat|fitnah) (jason|lucia|donie|donnie|doni|mendez)\b",
    # cara/tips dapat ending tertentu — menyingkap cabang cerita
    r"\b(cara|tips)\b.*\b(ending|trigger)\b",
    r"\b(god|good|true|happy|salah satu|salah|buanyak) ending\b",
    r"\b(ada|punya|berapa) \d+ (ending|pilihan)\b",
    r"\b(kalo|kalau|jangan|biar|supaya)\b.*\b(ending)\b",
    r"\bending (jason|lucia|donie|yang dia)\b",
    r"\bjangan (ngebunuh|milih|pilih|jemput|dikasihani|dihajar)\b",
    r"\bseharusnya kamu tidak usah\b",
    # trigger misi / kejadian penting
    r"\bmisi (bunuh|membunuh|akhir|terakhir)\b.*\b(trigger|ending)\b",
    r"\btrigger salah satu ending\b",
    r"\b(kumpulin|nabung) (2 ?juta|2jt)\b",
    r"\bkeluar (dari )?penjara\b",
    r"\bpas (dia|windah|jason|lucia|donie) \w+",
    r"\bternyata (dia|itu|windah|jason|lucia)\b",
    r"\b(di menit|di \d+:\d+|menit \d+)\b.*\b(ternyata|pas|dibunuh|mati|jebol|muncul)\b",
]
_POLA_BUKAN = [
    r"^jangan spoiler\s*[!.]*$",
    # pujian, reaksi, sapaan, semangat, promo part 2, saran tontonan
    r"\b(keren|mantap|bagus|seru|lucu|gokil|epik|semangat|top|cakep|gila |kangen|nonton|ikut|ikutin|tugas|sekolah|grafik|720p)\b",
    r"^(part|episode|ep|lanjutan) \d+",
    r"\b(bocil caper|buzzer|capernya|membership|saweria|adsense)\b",
    r"\b(aku|saya) (suka|senang|baru|benci|tunggu|nunggu)\b",
    r"^\d+:\d+(:\d+)? (parah|perfect|epic|adegan|lucu|gokil)\b",
]


_RE_TIMESTAMP = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*([a-z].*)?$")


def label_heuristik(teks: str) -> int:
    """Beri label 0/1/-1 dari satu teks komentar (versi ringan, tanpa stemming)."""
    t = " " + str(teks).lower() + " "
    if not t.strip():
        return -1
    for pola in _POLA_SPOILER:
        if re.search(pola, t):
            return 1
    for pola in _POLA_BUKAN:
        if re.search(pola, t):
            return 0
    # Timestamp polos ("1:48:00", "2:36:59 parah") = penanda momen, bukan isi.
    if _RE_TIMESTAMP.match(str(teks).strip()):
        return 0
    # Reaksi singkat tanpa isi cerita (<=4 kata, tanpa kata pemicu di atas) = 0.
    if len(t.split()) <= 5:
        return 0
    # Sisanya (>=6 kata, tanpa pola): perlakukan sebagai bukan spoiler —
    # keluhan/saran/cerita pribadi tanpa pola cerita. Lebih baik 0 daripada
    # membuang mayoritas data (dulu 277 ambigu).
    return 0


def main() -> None:
    """CLI: python -m src.pelabel_awal"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Pelabel heuristik awal (label_model, bukan manusia).")
    parser.add_argument("--input", type=Path, default=DATA_INTERIM / "komentar_bersih.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df["label_model"] = df["teks_bert"].apply(label_heuristik)
    logger.info("Distribusi label_model:\n%s", df["label_model"].value_counts().to_string())

    df["label"] = ""
    df["catatan"] = ""
    # Lembar label A/B tetap dibuat/diperbarui agar alur kesepakatan bisa dijalankan
    # nanti dengan label manusia sungguhan; kolom label_model ikut sebagai rujukan.
    lembar = df[["id", "teks", "label_model", "label", "catatan"]]
    for tag in ("a", "b"):
        lembar.to_csv(DATA_LABELED / f"lembar_label_{tag}.csv", index=False, encoding="utf-8-sig")

    # Sementara: label_final = label_model (ditandai jelas di kolom terpisah).
    df["label_final"] = df["label_model"]
    df["catatan"] = "PRELIMINAR: label_final berasal dari heuristik label_model, bukan manusia"
    df["sumber_label"] = "label_model_heuristik"

    out = DATA_LABELED / "komentar_berlabel.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("Tersimpan %s (%d baris, SEMUA PRELIMINAR)", out, len(df))


if __name__ == "__main__":
    main()
