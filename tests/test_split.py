"""Test split.py: reject invalid data, no label_model in src/."""
import subprocess

import pandas as pd
import pytest

from src.split import (
    ALL_VALID_IDS,
    SPOILER_EXAMPLE_IDS,
    TRAIN_VIDEO_IDS,
    VAL_VIDEO_IDS,
    TEST_VIDEO_IDS,
    validasi_data,
    buat_split,
)


def test_no_label_model_in_src():
    """grep -r 'label_model' src/ harus tidak menemukan apa pun."""
    result = subprocess.run(
        ["grep", "-r", "label_model", "src/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"label_model ditemukan di src/:\n{result.stdout}"


def test_video_ids_tidak_beririsan():
    """Train, val, test video IDs tidak boleh beririsan."""
    assert not (TRAIN_VIDEO_IDS & VAL_VIDEO_IDS), "Train dan val beririsan"
    assert not (TRAIN_VIDEO_IDS & TEST_VIDEO_IDS), "Train dan test beririsan"
    assert not (VAL_VIDEO_IDS & TEST_VIDEO_IDS), "Val dan test beririsan"


def test_validasi_reject_status_tidak_valid():
    """Validasi harus reject baris dengan status != 'valid'."""
    df = pd.DataFrame({
        "id": ["id1", "id2", "id3"],
        "video_id": ["Lve9p0C5rM4", "M15N9KPMHcE", "jh2ZJRbNmGY"],
        "label_final": [0, 1, 0],
        "status": ["valid", "tidak_valid_timestamp", "tidak_valid_kosong_atau_tautan"],
    })
    hasil = validasi_data(df)
    assert len(hasil) == 1
    assert hasil.iloc[0]["id"] == "id1"


def test_validasi_reject_label_kosong():
    """Validasi harus reject baris dengan label_final kosong atau NaN."""
    df = pd.DataFrame({
        "id": ["id1", "id2", "id3"],
        "video_id": ["Lve9p0C5rM4", "M15N9KPMHcE", "jh2ZJRbNmGY"],
        "label_final": [0, "", 1],
        "status": ["valid", "valid", "valid"],
    })
    hasil = validasi_data(df)
    assert len(hasil) == 2
    assert "id2" not in hasil["id"].values


def test_validasi_reject_spoiler_example():
    """Validasi harus reject baris dari spoiler_example (id 1-200)."""
    df = pd.DataFrame({
        "id": ["50", "id_real", "150"],
        "video_id": ["Lve9p0C5rM4", "M15N9KPMHcE", "jh2ZJRbNmGY"],
        "label_final": [1, 0, 1],
        "status": ["valid", "valid", "valid"],
    })
    hasil = validasi_data(df)
    assert len(hasil) == 1
    assert hasil.iloc[0]["id"] == "id_real"


def test_validasi_reject_video_tidak_di_tabel():
    """Validasi harus reject baris dengan video_id tidak di tabel pembagian."""
    df = pd.DataFrame({
        "id": ["id1", "id2", "id3"],
        "video_id": ["Lve9p0C5rM4", "INVALID_VIDEO", "M15N9KPMHcE"],
        "label_final": [0, 1, 0],
        "status": ["valid", "valid", "valid"],
    })
    hasil = validasi_data(df)
    assert len(hasil) == 2
    assert "id2" not in hasil["id"].values


def test_validasi_error_tanpa_video_id():
    """Validasi harus error jika kolom video_id tidak ada."""
    df = pd.DataFrame({
        "id": ["id1", "id2"],
        "label_final": [0, 1],
        "status": ["valid", "valid"],
    })
    with pytest.raises(ValueError, match="video_id"):
        validasi_data(df)


def test_buat_split_no_id_overlap():
    """Split harus tidak ada ID yang muncul di dua split berbeda."""
    df = pd.DataFrame({
        "id": [f"id{i}" for i in range(100)],
        "video_id": ["Lve9p0C5rM4"] * 100,
        "label_final": [0, 1] * 50,
        "status": ["valid"] * 100,
        "teks": ["teks"] * 100,
    })
    train, val, test = buat_split(df)
    
    id_train = set(train["id"])
    id_val = set(val["id"])
    id_test = set(test["id"])
    
    assert not (id_train & id_val), "ID ada di train dan val"
    assert not (id_train & id_test), "ID ada di train dan test"
    assert not (id_val & id_test), "ID ada di val dan test"
