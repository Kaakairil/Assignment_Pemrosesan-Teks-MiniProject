"""Ekstraksi komentar via YouTube Data API v3 (Tahap 1)."""
import argparse
import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import DATA_RAW, ROOT, TARGET_KOMENTAR_MENTAH, muat_env

logger = logging.getLogger(__name__)


def muat_key() -> str:
    """Baca YOUTUBE_API_KEY dari .env; hentikan jika kosong tanpa menampilkan nilai."""
    return muat_env("YOUTUBE_API_KEY")


def buat_layanan(api_key: str):
    """Buat objek layanan YouTube Data API v3."""
    return build("youtube", "v3", developerKey=api_key)


def ambil_semua_komentar(video_id: str, api_key: str, maks_komentar: int) -> list[dict]:
    """Ambil semua komentar utama (commentThreads.list), ikuti nextPageToken.

    Mengembalikan list mentah per halaman. Error kuota/commentsDisabled dibiarkan
    naik sebagai HttpError agar pemanggil CLI yang menangani.
    """
    youtube = buat_layanan(api_key)
    items: list[dict] = []
    page_token = None
    while len(items) < maks_komentar:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=page_token,
        ).execute()
        items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)
    return items[:maks_komentar]


def simpan_mentah(respons: dict, video_id: str) -> Path:
    """Simpan satu halaman respons API apa adanya ke data/raw/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    nama = f"komentar_{video_id}_{date.today().isoformat()}_{len(list(DATA_RAW.glob(f'komentar_{video_id}_{date.today().isoformat()}_*.json')))}.json"
    path = DATA_RAW / nama
    path.write_text(json.dumps(respons, ensure_ascii=False), encoding="utf-8")
    logger.info("Respons mentah disimpan: %s", path)
    return path


def id_anonim(author_channel_id: str, salt: str) -> str:
    """SHA-256 dari (salt + channel id), 16 karakter pertama."""
    return hashlib.sha256((salt + author_channel_id).encode("utf-8")).hexdigest()[:16]


def ke_dataframe(items: list[dict], salt: str) -> pd.DataFrame:
    """Ubah item API ke skema 6.2 (tanpa teks_klasik/teks_bert), anonimkan penulis."""
    baris = []
    for it in items:
        snip = it["snippet"]["topLevelComment"]["snippet"]
        baris.append(
            {
                "id": it["id"],
                "teks": snip.get("textDisplay", ""),
                "waktu": snip.get("publishedAt", ""),
                "jumlah_like": int(snip.get("likeCount", 0)),
                "balasan": int(it["snippet"].get("totalReplyCount", 0)),
                "id_anonim": id_anonim(snip.get("authorChannelId", {}).get("channelId", ""), salt),
            }
        )
    return pd.DataFrame(baris, columns=["id", "teks", "waktu", "jumlah_like", "balasan", "id_anonim"])


def proses_video(video_id: str, api_key: str, salt: str, maks_komentar: int) -> pd.DataFrame:
    """Ambil, simpan mentah per halaman, dan kembalikan DataFrame untuk satu video."""
    youtube = buat_layanan(api_key)
    items: list[dict] = []
    page_token = None
    halaman = 0
    while len(items) < maks_komentar:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=page_token,
            ).execute()
        except HttpError as e:
            alas = e.resp.status
            detail = ""
            try:
                detail = e.error_details[0].get("reason", "") if e.error_details else ""
            except Exception:
                detail = ""
            if alas == 403 and "commentsDisabled" in str(e.content):
                logger.warning("Komentar dinonaktifkan untuk %s — video dilewati.", video_id)
                return ke_dataframe(items, salt)
            if alas == 403 and "quotaExceeded" in str(e.content):
                logger.error("Kuota API habis di %s. Progres %d komentar dipertahankan.", video_id, len(items))
                return ke_dataframe(items, salt)
            logger.error("HttpError %s pada %s: %s", alas, video_id, detail)
            raise
        simpan_mentah(resp, video_id)
        halaman += 1
        items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        logger.info("%s: halaman %d, total %d komentar", video_id, halaman, len(items))
        if not page_token:
            break
        time.sleep(0.1)
    logger.info("%s: selesai, %d komentar dari %d halaman", video_id, len(items), halaman)
    return ke_dataframe(items[:maks_komentar], salt)


def main() -> None:
    """CLI: python -m src.ambil_komentar --video-id ID [--maks N]"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ambil komentar YouTube dan simpan mentah.")
    parser.add_argument("--video-id", default="", help="ID video YouTube (bisa beberapa, dipisah koma); default dari VIDEO_ID di .env")
    parser.add_argument("--maks", type=int, default=TARGET_KOMENTAR_MENTAH, help="Maks komentar per video")
    args = parser.parse_args()

    api_key = muat_key()
    salt = muat_env("HASH_SALT")

    if not args.video_id:
        args.video_id = muat_env("VIDEO_ID")

    gabungan = []
    for vid in [v.strip() for v in args.video_id.split(",") if v.strip()]:
        df = proses_video(vid, api_key, salt, args.maks)
        gabungan.append(df)

    hasil = pd.concat(gabungan, ignore_index=True)
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "komentar_mentah.csv"
    hasil.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Total %d komentar disimpan ke %s", len(hasil), path)


if __name__ == "__main__":
    main()
