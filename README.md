# Proyek Deteksi Spoiler — Komentar Replay GTA 6 (Windah Basudara)

Perbandingan dua metode klasifikasi biner untuk mendeteksi komentar spoiler pada
kolom komentar video replay YouTube Windah Basudara tentang GTA 6.

- **Metode A (baseline):** TF-IDF + Logistic Regression
- **Metode B (pembanding):** IndoBERT (`indobenchmark/indobert-base-p1`) fine-tuning

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Buat `.env` (jangan di-commit):

```
YOUTUBE_API_KEY=...
VIDEO_ID=id1,id2,...        # beberapa ID dipisah koma
HASH_SALT=string_acak_panjang
```

## Alur kerja (tahapan CLAUDE.md)

```bash
# Tahap 1 — ambil komentar (semua video di VIDEO_ID)
python -m src.ambil_komentar --video-id $VIDEO_ID --maks 1000

# Tahap 2 — preprocessing dua versi teks
python -m src.preprocessing

# Tahap 3 — buat lembar label A/B + subset kesepakatan 100
python -m src.buat_lembar_label

# (manual) kedua pelabel mengisi lembar_label_a.csv dan lembar_label_b.csv

# Tahap 4 — Cohen's kappa (berhenti jika < 0.6)
python -m src.kesepakatan

# Tahap 5 — gabung label_final dan split stratified
#   (jalankan setelah lembar digabung ke komentar_berlabel.csv dengan label_final)
python -m src.split

# Tahap 6 — Metode A
python -m src.metode_a

# Tahap 7 — Metode B (butuh GPU)
python -m src.metode_b

# Tahap 8 — evaluasi
python -m src.evaluasi

# Tahap 9 — analisis kesalahan
python -m src.analisis_kesalahan
```

## Struktur

Lihat CLAUDE.md bagian 4. Data mentah (`data/raw/`) tidak di-commit; yang
dibagikan adalah `data/labeled/` yang sudah dianonimkan (id_anonim = SHA-256
dari HASH_SALT + channel id, 16 karakter).

## Keputusan

- **PERINGATAN PRELIMINAR:** Hasil Metode A dan B yang tersimpan saat ini
  dilatih dengan **label heuristik** (`src/pelabel_awal.py`, kolom
  `label_model`), BUKAN label manusia. Sesuai CLAUDE.md, label otomatis tidak
  boleh disebut label manusia; Cohen's kappa tidak dihitung (dilewati, bukan
  dipalsukan). Ulangi pipeline dari `split` setelah lembar
  `lembar_label_a/b.csv` diisi manusia untuk hasil final.
- **Ketimpangan kelas ekstrem:** hanya ~31 komentar spoiler (1.1%) dari 2795
  — komentar atas di replay umumnya reaksi/timestamp/sapaan. Uji hanya
  berisi ~6 spoiler; metrik per-seed akan berisik. Dicatat sebagai
  keterbatasan utama laporan.
- **CPU untuk Metode B:** gate GPU di `metode_b` dilewati dengan flag
  `--pakai-cpu` atas permintaan pemilik ("lanjut sampai model A dan B jadi").
  Untuk laporan final, fine-tuning ulang di GPU (Colab) sangat disarankan.
- **numpy 2.x:** templat `requirements.txt` di CLAUDE.md memakai `numpy==1.*`,
  tetapi pyarrow/datasets modern butuh numpy≥2 — dinaikkan ke `numpy==2.*`.
- **Beberapa video:** CLAUDE.md menyebut satu video, tetapi pemilik memberikan
  10 tautan video live Windah Basudara. `ambil_komentar` menerima beberapa ID
  dipisah koma; setiap video dibatasi `--maks 1000` komentar mentah per video.
- **Kamus alay:** dibuat manual, konservatif (hanya penggantian yang jelas:
  singkatan umum, bentuk tawa `wkwk*` dinormalisasi ke `wkwk`, pengulangan
  huruf >2 dipangkas ke 2). Kata yang berisiko salah makna tidak dimasukkan.
- **Versi teks:** `teks_klasik` (stemming + tanpa stopword Sastrawi) untuk
  Metode A; `teks_bert` (bersih ringan, emoji dibiarkan) untuk Metode B —
  sesuai catatan perbandingan adil di rencana proyek.
- **Waktu pengambilan:** dicatat pada nama file mentah
  (`komentar_<VIDEO_ID>_<tanggal>_N.json`).
- **Metode B agregat:** evaluasi memakai konsensus rata-rata skor antar seed
  (ambang 0.5) untuk confusion matrix; per-seed metrik juga tersimpan.
- **Jumlah ending GTA 6:** TODO_VERIFIKASI — belum dicantumkan tanpa sumber resmi.
