"""Test fungsi preprocessing (fungsi murni, tanpa efek samping)."""
import pandas as pd

from src.preprocessing import (
    bersihkan_dasar,
    hapus_tidak_valid,
    hanya_timestamp,
    normalisasi_alay,
    teks_bert,
    teks_klasik,
    ulang_huruf,
)

KAMUS = {"gk": "tidak", "ga": "tidak", "bgt": "banget"}


def test_url_dan_mention_hilang():
    teks = "lihat https://x.com/abc dan @orang_oke yuk"
    hasil = bersihkan_dasar(teks)
    assert "http" not in hasil and "@orang_oke" not in hasil


def test_alay_ke_baku():
    assert normalisasi_alay("aku gk nonton", KAMUS) == "aku tidak nonton"
    assert "tidak" in teks_bert("aku ga suka", KAMUS)


def test_bangettt_jadi_banget():
    assert ulang_huruf("bangettt") == "banget"


def test_wkwkwk_dinormalisasi_bukan_dihapus():
    assert ulang_huruf("wkwkwkwk") == "wkwk"


def test_teks_bert_tidak_stemming():
    # "menangis" tidak boleh dipotong ke bentuk dasar di versi BERT
    hasil = teks_bert("aku menangis", KAMUS)
    assert "menangis" in hasil


def test_teks_klasik_stemming_dan_stopword():
    hasil = teks_klasik("aku menangis bgt", KAMUS)
    assert "aku" not in hasil.split()          # stopword hilang
    assert hasil.strip() != ""


def test_emoji_hilang_di_klasik_tetap_di_bert():
    assert "😭" not in teks_klasik("keren 😭", KAMUS)
    assert "😭" in teks_bert("keren 😭", KAMUS)


def test_hanya_timestamp_kasus_wajib():
    assert hanya_timestamp("1:30:30") is True
    assert hanya_timestamp("31:38 41:42 55:52") is True
    assert hanya_timestamp("1") is False
    assert hanya_timestamp("3") is False
    assert hanya_timestamp("bang misi nya seru 56:30") is False
    assert hanya_timestamp("") is False


def test_hapus_tidak_valid_menandai_status_tidak_menghapus():
    df = pd.DataFrame(
        {
            "teks": ["spoiler", "   ", "spoiler", "https://x.com 😭", "1:30:30", "56:30", "spoiler"],
            "teks_bert": ["spoiler", "   ", "spoiler", "https://x.com 😭", "1:30:30", "56:30", "spoiler"],
        }
    )
    hasil = hapus_tidak_valid(df)
    # Tidak ada baris dihapus; semua baris tetap ada dengan status.
    assert len(hasil) == len(df)
    assert list(hasil["status"]) == [
        "valid",
        "tidak_valid_kosong_atau_tautan",
        "duplikat",
        "tidak_valid_kosong_atau_tautan",
        "tidak_valid_timestamp",
        "tidak_valid_timestamp",
        "duplikat",
    ]
