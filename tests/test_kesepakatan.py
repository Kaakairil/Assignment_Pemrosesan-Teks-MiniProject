"""Test kesepakatan: kappa, pengecualian -1, error saat kosong."""
import pandas as pd
import pytest

from src.kesepakatan import hitung_kappa, muat_lembar


def test_hitung_kappa_sempurna():
    a = pd.DataFrame({"id": ["yt_1", "yt_2", "yt_3", "yt_4"], "label": [0, 1, 0, 1]})
    b = pd.DataFrame({"id": ["yt_1", "yt_2", "yt_3", "yt_4"], "label": [0, 1, 0, 1]})
    hasil = hitung_kappa(a, b, ["yt_1", "yt_2", "yt_3", "yt_4"])
    assert hasil["kappa"] == 1.0
    assert hasil["n"] == 4


def test_hitung_kappa_buang_minus1_dan_kosong():
    # -1 dan NaN dikeluarkan dari pasangan; hanya 2 pasangan valid tersisa.
    a = pd.DataFrame({"id": ["yt_1", "yt_2", "yt_3", "yt_4"], "label": [0, -1, 1, None]})
    b = pd.DataFrame({"id": ["yt_1", "yt_2", "yt_3", "yt_4"], "label": [0, 1, 1, 0]})
    hasil = hitung_kappa(a, b, ["yt_1", "yt_2", "yt_3", "yt_4"])
    assert hasil["n"] == 2
    assert hasil["kappa"] == 1.0


def test_hitung_kappa_error_semua_invalid():
    a = pd.DataFrame({"id": ["yt_1", "yt_2"], "label": [-1, None]})
    b = pd.DataFrame({"id": ["yt_1", "yt_2"], "label": [0, 0]})
    with pytest.raises(ValueError, match="pasangan"):
        hitung_kappa(a, b, ["yt_1", "yt_2"])


def test_muat_lembar_kolom_wajib(tmp_path):
    f = tmp_path / "lembar.csv"
    f.write_text("id,teks\nyt_1,teks\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        muat_lembar(f)
