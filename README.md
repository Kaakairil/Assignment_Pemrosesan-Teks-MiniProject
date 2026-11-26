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

# Tahap 5 — gabung label_final dan split per video (train #1-8, val #9-10, test #11)
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
- **2026-11-26 — Kappa Cohen = 0,580 (n=100) — DI BAWAH AMBANG 0,6.**
  Pelabel B menyelesaikan subset kesepakatan (96 bukan spoiler, 4 spoiler).
  Hanya 4 ketidaksesuaian: 2 kasus batas nyata (komentar peristiwa gameplay
  spesifik seperti "mati dimakan b2" dan "NPC teleport one punch" — A=1,
  B=0) dan 2 kasus ambigu (opini bug/curang; referensi karakter game lain).
  Hasil tertulis di hasil/kesepakatan.txt. Tindak lanjut SELESAI: 4 kasus
  dibahas dengan pemilik — 3 revisi label A (NPC teleport = G1 0; referensi
  karakter game lain = 0; 1 tetap: "mati dimakan b2" = S1 label A dipertahankan).
  Kappa ulang = **0,740** — LULUS ambang 0,6, pelabelan penuh dibuka.
  Catatan: label A pada 100 subset menunjukkan kesepakatan 98% dengan B.
- **2026-11-26 — Pelabelan A selesai (3.607 baris).** Mode cepat
  (ENTER=0) + triage sekilas, diikuti audit kata kunci berbasis panduan:
  90 baris berlabel 0 bersinyal kuat dicek acak, 14 baris risiko tinggi
  (ternyata/twist/ending) direview satu per satu — semuanya konsisten
  dengan panduan v2.1. Kesimpulan penting untuk pelabel: kata "ending" di
  komentar replay bisa berarti AKHIR STREAM, bukan ending game — label
  mengikuti isi. Distribusi: 3.468 bukan spoiler (96,1%), 94 spoiler
  (2,6%), 45 ambigu (1,2%). Proporsi spoiler jauh di bawah pilot (6,4%)
  karena triage kilat; dicatat sebagai risiko utama laporan.
- **2026-11-26 — Tes pemahaman pelabel A (Lampiran B): LULUS 7/8.**
  Pelabel A melabeli 8 baris dari contoh_200_spoiler.csv tanpa melihat kolom
  label; skor terhadap panduan 2.1 = 7/8 (ambang 4/5). Satu-satunya meleset:
  id 47 (mekanik rambut/jenggot real-time, dilabeli 1 padahal G1 = 0) —
  konflik G1 yang sudah diprediksi; label file contoh memang bertentangan
  dengan panduan pada kategori Bocoran Mekanik.
- **2026-11-26 — Klarifikasi G1 (easter egg/misteri).** Komentar yang
  mengungkap lokasi/keberadaan easter egg atau misteri dunia (bigfoot, UFO,
  portal, topi Arthur) TANPA dampak ke alur/nasib karakter = 0. Alasan:
  definisi spoiler proyek adalah spoiler cerita, bukan "spoiler pengalaman
  menemukan". Ditambahkan ke panduan sebelum pelabelan penuh (panduan belum
  dibekukan untuk pelabel B); tidak ada label lama yang terdampak.
- **2026-11-26 — Panduan label v2.1 disetujui dan dikunci.** Pemilik
  menyetujui `panduan/panduan_label.md` (aturan S1–S6, G1, T1, R1, contoh S5
  tips+konsekuensi). Mulai sekarang perubahan panduan wajib dicatat ulang dan
  memicu pelabelan ulang data terdampak.
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
- **Training Metode B di GPU lokal (RTX 3050 Laptop 4GB).** Percobaan Google
  Colab gagal berulang (clone ganda, sel korup, runtime terputus, dan satu
  insiden training pada data lama yang tidak valid), sehingga notebook Colab
  dihapus dan training dipindah ke laptop dengan torch cu128. Gate GPU di
  `metode_b` kini lewat karena CUDA tersedia. `save_total_limit=1`,
  batch train 4 × grad-accum 2, eval batch 8 (batas VRAM 4GB). Tiga run
  penuh mengalami crash pada fase akhir (save/predict), sehingga prediksi
  per seed direkonstruksi manual dari checkpoint terbaik (seed42→ckpt-798,
  seed43→ckpt-1064, seed44→ckpt-798).
- **Konsensus seed dihapus (panduan 2.1).** Metode B dilaporkan sebagai
  rata-rata ± simpangan baku antar tiga seed (42/43/44); tidak ada prediksi
  gabungan seed dan tidak ada seed dipilih berdasarkan data uji. McNemar
  dijalankan per seed.
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
- **Jumlah ending GTA 6:** TODO_VERIFIKASI — belum dicantumkan tanpa sumber resmi.
