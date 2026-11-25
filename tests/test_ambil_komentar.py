"""Test ekstraksi komentar dengan data tiruan (mock), tanpa panggilan API sungguhan."""
import json

import pandas as pd
import pytest

from src.ambil_komentar import id_anonim, ke_dataframe, simpan_mentah
from src.config import DATA_RAW


def _item(cid: str, teks: str, channel: str, like: int = 0, balasan: int = 0) -> dict:
    return {
        "id": cid,
        "snippet": {
            "totalReplyCount": balasan,
            "topLevelComment": {
                "id": cid,
                "snippet": {
                    "textDisplay": teks,
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "likeCount": like,
                    "authorChannelId": {"channelId": channel},
                },
            },
        },
    }


def test_ke_dataframe_skema_dan_anonim():
    items = [_item("c1", "keren banget", "UC123", like=5, balasan=2)]
    df = ke_dataframe(items, salt="SALT")
    assert list(df.columns) == ["id", "teks", "waktu", "jumlah_like", "balasan", "id_anonim"]
    assert df.loc[0, "id_anonim"] != "UC123"
    assert len(df.loc[0, "id_anonim"]) == 16


def test_id_anonim_deterministik_dan_bergantung_salt():
    a = id_anonim("UC123", "salt1")
    assert a == id_anonim("UC123", "salt1")
    assert a != id_anonim("UC123", "salt2")


def test_simpan_mentah(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ambil_komentar.DATA_RAW", tmp_path)
    respons = {"items": [_item("c1", "teks", "UC1")]}
    path = simpan_mentah(respons, "vid123")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["items"][0]["id"] == "c1"
