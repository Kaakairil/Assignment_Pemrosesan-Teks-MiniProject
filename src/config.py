"""Konstanta, path, dan seed global proyek. Semua angka ajaib berasal dari sini."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_LABELED = ROOT / "data" / "labeled"
PANDUAN = ROOT / "panduan"
KAMUS_PATH = ROOT / "kamus" / "normalisasi_alay.csv"
HASIL = ROOT / "hasil"
MODELS = ROOT / "models"

SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1                  # diambil dari data latih
MIN_KAPPA = 0.6
MAX_LEN_BERT = 128
MODEL_BERT = "indobenchmark/indobert-base-p1"   # cek lisensi dan ganti bila perlu
SEEDS_BERT = [42, 43, 44]
N_BOOTSTRAP = 1000              # resample bootstrap untuk CI F1 pada test
ALPHA_CI = 0.05                 # tingkat signifikansi CI (0.05 -> CI 95%)
MIN_KOMENTAR_BERSIH = 500
TARGET_KOMENTAR_MENTAH = 1000

LABEL_SPOILER = 1
LABEL_BUKAN = 0
LABEL_AMBIGU = -1

load_dotenv(ROOT / ".env")


def _daftar_video(kunci: str) -> list[str]:
    """Baca daftar video ID dari .env (dipisah koma), tanpa menampilkan nilainya."""
    nilai = os.getenv(kunci, "").strip()
    if not nilai:
        raise SystemExit(f"Variabel {kunci} kosong di .env. Isi sesuai dokumen proyek (docs/update) bagian 2.")
    return [v.strip() for v in nilai.split(",") if v.strip()]


TRAIN_VIDEO_IDS = _daftar_video("TRAIN_VIDEO_IDS")
VAL_VIDEO_IDS = _daftar_video("VAL_VIDEO_IDS")
TEST_VIDEO_IDS = _daftar_video("TEST_VIDEO_IDS")

# Tidak boleh ada video yang muncul di dua peran.
assert not (set(TRAIN_VIDEO_IDS) & set(VAL_VIDEO_IDS)), "TRAIN dan VAL beririsan"
assert not (set(TRAIN_VIDEO_IDS) & set(TEST_VIDEO_IDS)), "TRAIN dan TEST beririsan"
assert not (set(VAL_VIDEO_IDS) & set(TEST_VIDEO_IDS)), "VAL dan TEST beririsan"

SEMUA_VIDEO_IDS = TRAIN_VIDEO_IDS + VAL_VIDEO_IDS + TEST_VIDEO_IDS


def muat_env(kunci: str) -> str:
    """Baca variabel .env; hentikan dengan pesan jelas jika kosong (tanpa menampilkan nilai)."""
    nilai = os.getenv(kunci, "").strip()
    if not nilai:
        raise SystemExit(
            f"Variabel {kunci} kosong di .env. Isi dulu file .env (jangan di-commit)."
        )
    return nilai
