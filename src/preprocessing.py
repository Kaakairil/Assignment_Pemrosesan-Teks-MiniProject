"""Pembersihan dan preprocessing dua versi teks (Tahap 2)."""
import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from src.config import KAMUS_PATH, ROOT

logger = logging.getLogger(__name__)

# Dibuat sekali di level modul, bukan di dalam loop.
_STEMMER = StemmerFactory().create_stemmer()
# Daftar default Sastrawi tidak memuat beberapa pronomina umum (mis. "aku").
_TAMBAHAN_STOPWORD = ["aku", "kamu", "dia", "mereka", "kita", "banget"]
_SW_FACTORY = StopWordRemoverFactory()
_STOPWORD_REMOVER = StopWordRemover(
    ArrayDictionary(_SW_FACTORY.get_stop_words() + _TAMBAHAN_STOPWORD)
)

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_KONTROL = re.compile(r"[\x00-\x1f\x7f]")
_RE_SPASI = re.compile(r"\s+")
_RE_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F02F"
    "\U00002600-\U000026FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF️‍]+"
)
# Pengulangan huruf lebih dari dua dipangkas jadi satu (contoh wajib dari
# CLAUDE.md: "bangettt" -> "banget"), kecuali pola wkwk yang dinormalisasi ke "wkwk".
_RE_ULANG = re.compile(r"(.)\1{2,}")
_RE_WKW = re.compile(r"\b(?:wk){2,}\b")

# Status baris (kolom `status`, baris tidak dihapus agar bisa diaudit).
STATUS_VALID = "valid"
STATUS_TIMESTAMP = "tidak_valid_timestamp"
STATUS_KOSONG = "tidak_valid_kosong_atau_tautan"
STATUS_DUPLIKAT = "duplikat"


def hanya_timestamp(teks: str) -> bool:
    """True jika teks hanya berisi angka, titik dua, dan spasi — dan punya minimal satu titik dua."""
    t = teks.strip()
    return bool(re.fullmatch(r"[\d:\s]+", t)) and ":" in t


def muat_kamus(path: Path = KAMUS_PATH) -> dict[str, str]:
    """Muat kamus normalisasi alay dari CSV (kolom: kata_alay,kata_baku)."""
    df = pd.read_csv(path, encoding="utf-8")
    return dict(zip(df["kata_alay"].astype(str).str.strip(), df["kata_baku"].astype(str).str.strip()))


def bersihkan_dasar(teks: str) -> str:
    """Lowercase, hapus URL, mention, kontrol, dan whitespace berlebih."""
    teks = str(teks)
    teks = _RE_URL.sub(" ", teks)
    teks = _RE_MENTION.sub(" ", teks)
    teks = _RE_KONTROL.sub(" ", teks)
    teks = teks.lower()
    return _RE_SPASI.sub(" ", teks).strip()


def normalisasi_alay(teks: str, kamus: dict[str, str]) -> str:
    """Ganti token per token berdasarkan kamus, pemisahan spasi."""
    return " ".join(kamus.get(tok, tok) for tok in teks.split())


def ulang_huruf(teks: str) -> str:
    """Pangkas pengulangan huruf >2 jadi satu; 'wkwkwk' dinormalisasi jadi 'wkwk'."""
    teks = _RE_WKW.sub("wkwk", teks)
    return _RE_ULANG.sub(r"\1", teks)


def teks_klasik(teks: str, kamus: dict[str, str]) -> str:
    """Versi Metode A: dasar + hapus emoji + normalisasi + hapus stopword + stemming."""
    teks = bersihkan_dasar(teks)
    teks = _RE_EMOJI.sub(" ", teks)
    teks = normalisasi_alay(teks, kamus)
    teks = ulang_huruf(teks)
    teks = _STOPWORD_REMOVER.remove(teks)
    return _STEMMER.stem(teks).strip()


def teks_bert(teks: str, kamus: dict[str, str]) -> str:
    """Versi Metode B: dasar + normalisasi + ulang huruf. Tanpa stemming/stopword."""
    teks = bersihkan_dasar(teks)
    teks = normalisasi_alay(teks, kamus)
    return ulang_huruf(teks).strip()


def _hanya_emoji_atau_tautan(teks: str) -> bool:
    """True jika teks kosong, hanya emoji, atau hanya tautan."""
    sisa = _RE_URL.sub(" ", teks)
    sisa = _RE_EMOJI.sub(" ", sisa)
    return not sisa.strip()


def hapus_tidak_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Tandai `status` tiap baris; baris TIDAK dihapus agar jumlah bisa diaudit.

    Kategori status: valid, tidak_valid_timestamp, tidak_valid_kosong_atau_tautan,
    duplikat. Deteksi memakai teks asli (`teks`), bukan teks_bert/teks_klasik,
    karena pemrosesan bisa menghapus titik dua dari timestamp.
    """
    awal = len(df)
    teks_asli = df["teks"].astype(str)

    # urutan prioritas: kosong/tautan -> timestamp -> duplikat (baris pertama valid)
    status = pd.Series(STATUS_VALID, index=df.index)
    kosong = df["teks_bert"].astype(str).str.strip().eq("") | teks_asli.apply(
        _hanya_emoji_atau_tautan
    )
    status[kosong] = STATUS_KOSONG

    timestamp = teks_asli.map(hanya_timestamp) & (status == STATUS_VALID)
    status[timestamp] = STATUS_TIMESTAMP

    dup = df.duplicated(subset="teks_bert", keep="first") & (status == STATUS_VALID)
    status[dup] = STATUS_DUPLIKAT

    df = df.copy()
    df["status"] = status
    logger.info("Status: %s (dari %d baris)", status.value_counts().to_dict(), awal)
    return df.reset_index(drop=True)


def proses_csv(input_path: Path, kamus: dict[str, str], output_path: Path) -> pd.DataFrame:
    """Baca CSV mentah, buat teks_klasik dan teks_bert, buang tidak valid, simpan."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info("Memproses %d baris dari %s", len(df), input_path)
    df["teks_klasik"] = df["teks"].apply(lambda t: teks_klasik(t, kamus))
    df["teks_bert"] = df["teks"].apply(lambda t: teks_bert(t, kamus))
    df = hapus_tidak_valid(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Tersimpan: %s (%d baris)", output_path, len(df))
    return df


def main() -> None:
    """CLI: python -m src.preprocessing --input ... --output ..."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Preprocessing komentar jadi dua versi teks.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "interim" / "komentar_mentah.csv")
    parser.add_argument("--kamus", type=Path, default=KAMUS_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "interim" / "komentar_bersih.csv")
    args = parser.parse_args()

    kamus = muat_kamus(args.kamus)
    proses_csv(args.input, kamus, args.output)


if __name__ == "__main__":
    main()
