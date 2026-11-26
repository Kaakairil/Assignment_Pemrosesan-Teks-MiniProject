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

- **2026-11-26 — Data lama tidak dipakai.** Seluruh 2.795 komentar lama
  dipindahkan ke `data/legacy/` dan tidak dibaca pipeline. Alasan: label
  `label_final` lama berasal dari heuristik `src/pelabel_awal.py` (kolom
  `label_model`), bukan pelabel manusia. Hasil evaluasi lama (F1 Metode A
  0,667; Metode B 0,665 ± 0,049) dinyatakan tidak valid dan tidak dilaporkan
  sebagai hasil akhir. Pelabelan manusia dimulai dari nol.
- **2026-11-26 — `src/pelabel_awal.py` dihapus.** Modul pelabel heuristik
  dihapus dari pipeline (`git rm`); kolom `label_model` dihapus dari semua
  CSV. Tes `tests/test_split.py::test_no_label_model_in_src` mencegah kolom
  itu muncul lagi di `src/`.
- **2026-11-26 — Pilot (Bagian 5).** 157 komentar pertama dilabeli manual
  (lembar_label_a): 10 spoiler = 6,4%, di atas ambang ~5%. Keputusan: sampling
  acak komentar teratas per episode cukup; tidak perlu sampling terarah.
  Catatan: 2 label tip mekanik awalnya salah dilabeli 1, direvisi ke 0
  sesuai aturan G1 setelah pelabel mengingat aturan. Basis 10 spoiler tipis
  untuk estimasi proporsi (CI ±3–5 poin), dicatat sebagai keterbatasan.
- **2026-11-26 — `data/spoiler_example/` tidak masuk dataset.** Isi folder
  (termasuk `contoh_200_spoiler.csv`) hanya untuk latihan pelabel dan tes
  kode. Catatan: label di `contoh_200_spoiler.csv` BERTENTANGAN dengan aturan
  G1 (tips dan mekanik dilabeli `spoiler`, padahal G1 menyatakan mekanik
  tanpa dampak cerita bukan spoiler), dan label-nya tidak berasal dari
  pelabel manusia. Tidak dipakai untuk train/val/test.
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
