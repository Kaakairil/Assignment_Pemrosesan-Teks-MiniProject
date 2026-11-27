# Laporan Mini-Proyek: Deteksi Spoiler Komentar Replay GTA 6 (Windah Basudara)

Mata kuliah Pemrosesan Teks, semester 5. Tanggal laporan: 2026-11-27.

## 1. Pendahuluan

Komentar pada video arsip (replay) live streaming sering memuat spoiler dari
penonton yang sudah menonton lebih jauh. Penonton yang belum menyaksikan
episode tersebut dapat terganggu. Proyek ini membandingkan dua metode
klasifikasi biner untuk mendeteksi komentar spoiler pada kolom komentar seri
"GTA 6 Gameplay" Windah Basudara episode #1 sampai #11:

- **Pertanyaan penelitian 1:** Seberapa besar perbedaan F1 dan recall kelas
  spoiler antara IndoBERT dan TF-IDF + Logistic Regression?
- **Pertanyaan penelitian 2:** Pada jenis komentar apa kedua metode paling
  sering salah, dan apakah kesalahan itu berasal dari model atau dari label?

- **Metode A (baseline):** TF-IDF + Logistic Regression
- **Metode B (pembanding):** IndoBERT (`indobenchmark/indobert-base-p1`)
  yang di-fine-tune
- **Label:** `1` = spoiler, `0` = bukan spoiler, `-1` = ambigu (dibuang dari
  latih, validasi, dan uji)

**Hipotesis awal:** IndoBERT memberi F1 kelas spoiler lebih tinggi, terutama
pada komentar bahasa campuran dan sindiran.

## 2. Tinjauan pustaka

- **TF-IDF dan regresi logistik untuk klasifikasi teks** adalah baseline
  baku pada tugas klasifikasi dokumen pendek (Joachims, 1998 — *Text
  Categorization with Support Vector Machines: Learning with Many Relevant
  Features*; reformulasi TF-IDF klasik dari Salton & Buckley, 1988 — *Term-
  weighting approaches in automatic text retrieval*). Logistic regression
  dengan class weight `balanced` umum dipakai pada data tak seimbang.
- **Fine-tuning model bahasa pralatih** (Devlin et al., 2019 — *BERT:
  Pre-training of Deep Bidirectional Transformers for Language
  Understanding*) terbukti unggul pada banyak tugas klasifikasi dengan data
  terbatas. Untuk bahasa Indonesia, **IndoBERT** (Koto, Rahmani & Aji,
  2020 — *IndoBERT: A Pretrained Language Model for Indonesian*, riset
  IndoNLP) adalah model pralatih publik yang dilatih pada korpus bahasa
  Indonesia dan tersedia di Hugging Face sebagai
  `indobenchmark/indobert-base-p1`.
- Klasifikasi komentar YouTube berbahasa Indonesia informal (bahasa gaul,
  singkatan, bahasa campuran) menambah tantangan normalisasi; kamus
  normalisasi alay manual dipakai sebagai mitigasi (lihat §4).

## 3. Data

### 3.1 Sumber dan pengambilan

- Komentar diambil dari YouTube Data API (`commentThreads`) pada
  **2026-11-25** untuk episode #1–#10 dan **2026-11-26** untuk episode #11,
  dari kanal Windah Basudara, seri "GTA 6 Gameplay" episode #1–#11. Semua
  video adalah arsip live streaming. Metadata views dicatat pada
  2026-11-26 di `hasil/pemilihan_video.md` dan **berubah seiring waktu**;
  contoh: episode #1 berviews 16.017.735 per tanggal tersebut.
- Komentar mentah disimpan per halaman di `data/raw/` (tidak di-commit).
- Tidak ada data sintetis. Data lama (2.795 komentar berlabel heuristik)
  dipindahkan ke `data/legacy/` dan **tidak dipakai pipeline**; hasil
  evaluasi lama (F1 A 0,667; B 0,665 ± 0,049) dinyatakan tidak valid.

### 3.2 Pelabelan

- Panduan label v2.1 (`panduan/panduan_label.md`): aturan S1–S6, G1, T1, R1.
  Definisi spoiler: mengungkap isi cerita atau konsekuensi cerita yang belum
  diketahui pembaca yang belum menamatkan game — alur, akhir, nasib
  karakter, twist, peristiwa penting, dan mekanik yang menentukan salah
  satunya (S5). Jumlah ending GTA 6 **tidak** menjadi dasar label dan
  belum terverifikasi (`TODO_VERIFIKASI`).
- Dua pelabel manusia; kesepakatan diukur pada subset 100 komentar.
  Kappa Cohen awal 0,580 (di bawah ambang 0,6) — 4 kasus dibahas, 3 revisi
  label, kappa ulang **0,740 (LULUS)**, lalu pelabelan penuh dibuka.
- Hasil pelabelan: 3.562 komentar valid. Distribusi: 3.470 bukan spoiler
  (97,4%), 92 spoiler (2,6%). Label ambigu (-1) dibuang.
- Semua label berasal dari pelabel manusia; tidak ada label otomatis yang
  dihitung sebagai label manusia.

### 3.3 Pembagian data (per video, bukan acak)

Pembagian mengikuti urutan episode untuk mencegah kebocoran antar split
(komentar episode lanjutan sering membahas kejadian episode sebelumnya):
- **Latih:** episode #1–#8 → 2.126 komentar, 50 spoiler (2,4%)
- **Validasi:** episode #9–#10 → 539 komentar, 20 spoiler (3,7%)
- **Uji:** episode #11 (`5m-Xi8lgvWY`) → 897 komentar, 22 spoiler (2,5%)

Data uji **hanya dievaluasi satu kali**, setelah hyperparameter dan model
dikunci pada validasi.

## 4. Metodologi

### 4.1 Pra-pemrosesan

- **Kamus alay manual** (`kamus/normalisasi_alay.csv`), konservatif: hanya
  penggantian yang jelas maknanya; bentuk tawa `wkwk*` dinormalisasi;
  pengulangan huruf >2 dipangkas.
- Dua versi teks untuk perbandingan adil:
  - `teks_klasik`: stemming + tanpa stopword (Sastrawi) — untuk Metode A.
  - `teks_bert`: pembersihan ringan, emoji dibiarkan — untuk Metode B.
- `MAX_LEN_BERT = 128`; proporsi token terpotong dicatat saat pelatihan.

### 4.2 Metode A — TF-IDF + Logistic Regression

- Pipeline: `TfidfVectorizer(min_df=2, sublinear_tf=True)` +
  `LogisticRegression(class_weight="balanced", max_iter=1000)`.
- Grid: ngram (1,1)/(1,2) × max_features None/10.000 × C 0,1/1/10
  (12 kombinasi). **Pemilihan dengan data validasi** (F1 kelas spoiler),
  bukan StratifiedKFold acak — komentar dari video yang sama bisa bocor
  antar lipatan.
- Hasil pemilihan: **ngram (1,2), max_features None, C = 0,1**
  (F1 validasi 0,393). Model dilatih ulang pada train, lalu dievaluasi
  **sekali** pada test.

### 4.3 Metode B — IndoBERT fine-tuning

- Model: `indobenchmark/indobert-base-p1`, klasifikasi 2 label,
  `MAX_LEN_BERT = 128`.
- Hyperparameter terkunci: learning rate 2e-5, 4 epoch, batch efektif 8
  (train batch 4 × akumulasi gradien 2 — dibatasi VRAM RTX 3050 4GB),
  weight decay 0,01. Pemilihan checkpoint terbaik per seed berdasarkan F1
  validasi per epoch (early stopping semantik via `load_best_model_at_end`).
- Bobot kelas dihitung dari data latih saja, diterapkan pada
  cross-entropy (`TrainerBobot`).
- **Tiga seed** (42, 43, 44) dijalankan penuh; prediksi uji disimpan per
  seed (`hasil/prediksi_metode_b_seed*.csv`). Tidak ada "konsensus seed":
  tidak ada prediksi gabungan seed dan tidak ada seed yang dipilih
  berdasarkan data uji.
- Pelatihan dilakukan lokal di GPU RTX 3050 Laptop 4GB (torch 2.11.0+cu128)
  setelah percobaan Google Colab gagal berulang (detail di README bagian
  Keputusan).

### 4.4 Evaluasi

- Metrik utama: precision, recall, F1 kelas spoiler; macro F1 dan accuracy
  sebagai pendukung.
- Metode B dilaporkan sebagai **rata-rata ± simpangan baku antar tiga
  seed**.
- **CI 95% bootstrap** (N=1.000 resample baris test) untuk F1 kelas
  spoiler: A langsung; B dari rata-rata F1 per seed per resample.
- **Uji McNemar** A vs B pada prediksi uji yang sama, dijalankan per seed.

## 5. Hasil

### 5.1 Tabel metrik (data uji, 897 komentar, 22 spoiler)

| Metrik | A (TF-IDF + LogReg) | B (IndoBERT, mean ± std, 3 seed) |
|---|---|---|
| Precision spoiler | 0,266 | 0,605 |
| Recall spoiler | 0,773 | 0,364 ± 0,164 |
| **F1 spoiler** | **0,395** | **0,435 ± 0,120** |
| Macro F1 | 0,682 | 0,712 |
| Accuracy | 0,942 | 0,978 |

F1 per seed B: seed 42 = 0,345 (P 0,714 / R 0,227); seed 43 = 0,571
(P 0,600 / R 0,545); seed 44 = 0,389 (P 0,500 / R 0,318).

### 5.2 CI 95% bootstrap F1 spoiler

| Metode | F1 | CI 95% |
|---|---|---|
| A | 0,395 | [0,256, 0,520] |
| B | 0,435 | [0,260, 0,597] |

Kedua interval **saling tumpang tindih**.

### 5.3 McNemar (A vs B, per seed)

| Seed | Statistik | P-value |
|---|---|---|
| 42 | 17,96 | 0,000023 |
| 43 | 21,78 | 0,000003 |
| 44 | 15,57 | 0,000079 |

Semua p < 0,001: pola kesalahan kedua metode berbeda signifikan (kedua
metode tidak salah pada baris yang sama), tetapi ini tidak mengatakan
mana yang lebih baik.

### 5.4 Confusion matrix

- A: [[828, 47], [5, 17]] — 47 false positive, 5 false negative.
- B seed 42: precision tinggi (0,714) tetapi hanya menemukan 5 dari 22
  spoiler; seed 43 menemukan 12 dari 22. Variasi antar seed besar.

## 6. Interpretasi (Bagian 9.4 panduan perbaikan)

1. **Selisih F1 A vs B (0,040) lebih kecil dari simpangan baku antar seed
   B (0,120)**, dan CI 95% kedua metode saling tumpang tindih. Sesuai
   kriteria panduan: **perbedaan kinerja tidak terdeteksi** pada data uji
   ini. Hipotesis awal "IndoBERT lebih baik" tidak terkonfirmasi — tetapi
   juga tidak terbantahkan; data uji terlalu kecil untuk membedakan.
2. **Hasil hanya dari satu episode uji (#11)** dan bergantung pada jumlah
   spoiler yang kecil (22 komentar). Satu komentar yang berubah label
   memindahkan F1 sekitar 0,02–0,04 poin. Metrik per-seed B berisik
   (0,345–0,571) dan kesimpulan per-seed tidak boleh ditarik.
3. Trade-off karakter: A recall-tinggi/precision-rendah (memangkas dengan
   class weight balanced; 47 FP dari 875 non-spoiler), B cenderung
   sebaliknya dan sangat bervariasi antar seed. McNemar signifikan
   menunjukkan kedua metode salah pada komentar yang berbeda —
   peluang ensemble ada, tetapi di luar cakupan proyek.

## 7. Analisis kesalahan (Bagian 10)

`hasil/analisis_kesalahan.csv` berisi 64 baris salah klasifikasi (gabungan
A dan B seed 42, berdasarkan `id`): A — 5 FN, 47 FP; B — 17 FN, 2 FP.
Kolom `pola` dan `jenis_galat` (galat label vs galat model) diisi manual
oleh pemilik proyek dengan kategori: sindiran, bahasa_campuran,
teks_pendek, hanya_nama_karakter, menyebut_ending_tanpa_detail,
tips_mekanik, lainnya.

Pengamatan awal dari contoh kesalahan (belum lengkap sampai kolom diisi):

- **FN A (5)**: komentar saran hubungan Jason–Lucia yang menyebut "ending"
  sebagai konsekuensi (S6). Model leksikal tidak mengaitkan frasa
  bermakna campuran.
- **FP A (47)**: dominan komentar yang menyebut kata "ending" sebagai
  **akhir stream**, bukan ending game — jebakan semantik yang juga ditemukan
  saat audit pelabelan. Contoh: "kecewa ending", "kampret ending",
  "good byee bugatti".
- **FN B (17)**: pola serupa dengan FN A (saran hubungan + kata "ending"),
  ditambah teks pendek dan bahasa campuran informal.
- **FP B (2)**: komentar panjang yang merinci mekanik/misi secara detail.

Pemisahan galat label vs galat model menunggu pengisian manual; galat
label tidak dihitung sebagai kegagalan model.

## 8. Keterbatasan

1. **Satu kanal, satu seri, satu episode uji.** Generalisasi ke kanal atau
   game lain tidak didukung data ini.
2. **Jumlah spoiler kecil di uji (22 dari 897, 2,5%).** CI F1 lebar
   (lebar ~26 poin persen untuk A) dan metrik sangat sensitif terhadap
   beberapa komentar.
3. **Ketidakseimbangan kelas ekstrem (2,6% spoiler)** membuat F1 kelas
   spoiler secara inheren sulit; accuracy ~0,94–0,98 mencerminkan kelas
   mayoritas, bukan kinerja deteksi spoiler.
4. **Variasi antar seed B besar (std 0,120)** — setara atau lebih besar
   dari selisih antar metode. Tiga seed minimum, tetapi tidak cukup untuk
   estimasi stabil.
5. **Label subjektif.** Batas spoiler/bukan pada komentar replay
   (mis. peristiwa gameplay spesifik, S5/S6) adalah keputusan panduan,
   bukan kebenaran objektif. Kappa 0,740 menunjukkan kesepakatan kuat
   tetapi tidak sempurna.
6. **Komentar episode #10–#11 diambil beberapa jam setelah live** —
   distribusi komentar masih berkembang; tidak seperti episode awal yang
   sudah stabil.
7. **Klaim jumlah ending GTA 6 tidak diverifikasi** (`TODO_VERIFIKASI`)
   dan tidak dipakai sebagai dasar label.

## 9. Etika

1. **Anonimisasi.** Tidak ada nama pengguna, ID penulis asli, atau tautan
   profil di data yang dibagikan (`data/labeled/`, `hasil/`). Identitas
   diganti `id_anonim` = 16 karakter pertama SHA-256 dari (HASH_SALT + ID
   channel penulis); salt hanya di `.env` yang tidak di-commit.
2. **Tujuan.** Komentar hanya untuk penelitian mata kuliah ini, tidak
   ditampilkan ulang dengan identitas, tidak dipublikasikan ke pihak lain.
3. **Subjektivitas.** Sifat spoiler bersifat subjektif; didefinisikan
   eksplisit dalam panduan v2.1 dan diukur kesepakatannya (kappa).
4. **Syarat layanan.** Pengambilan hanya lewat YouTube Data API resmi.
   Tidak ada scraping Instagram (dilarang panduan proyek).
5. **Sumber terbuka.** Kode dan data anonim dibagikan untuk reproduksi
   akademik; `data/raw/`, `.env`, dan `models/` tidak di-commit.

## 10. Kesimpulan

1. Kedua metode dapat dibangun end-to-end dengan tata kelola ketat: split
   per video, pemilihan hyperparameter di validasi, uji sekali, tiga seed
   tanpa konsensus, CI bootstrap, dan McNemar per seed.
2. **Perbedaan F1 antara Metode A (0,395) dan Metode B (0,435 ± 0,120)
   tidak terdeteksi** — selisih lebih kecil dari simpangan baku antar seed
   dan CI saling tumpang tindih. Dengan 22 spoiler di satu episode uji,
   eksperimen ini tidak cukup berdaya untuk membedakan kedua metode.
3. McNemar signifikan (p < 0,001 semua seed) menunjukkan pola kesalahan
   berbeda: A meleset sedikit (5 FN) tapi salah menandai banyak (47 FP,
   kebanyakan kata "ending" = akhir stream); B lebih selektif dan
   bervariasi antar seed.
4. Untuk kerja lanjutan: episode uji tambahan (dengan jumlah spoiler lebih
   besar), seed lebih banyak, dan penanganan eksplisit kata "ending"
   (konteks akhir stream vs ending game) adalah tiga perbaikan dengan
   dampak terbesar.

## Lampiran A — Panduan label (ringkas)

Panduan penuh: `panduan/panduan_label.md` v2.1. Ringkasan aturan:

- **S1–S4:** alur, akhir, nasib karakter, twist/peristiwa penting = spoiler.
- **S5:** mekanik yang menentukan salah satu dari S1–S4 = spoiler;
  mekanik murni (tips tanpa konsekuensi cerita) = bukan.
- **S6:** keputusan pemain yang menentukan ending (mis. hubungan
  Jason–Lucia) — komentar yang menyebut konsekuensinya = spoiler.
- **G1:** easter egg/misteri dunia tanpa dampak alur/nasib = bukan spoiler.
- **T1, R1:** aturan teks dan repetisi. **-1 (ambigu)** bila tidak yakin,
  dengan catatan.
- Label mengikuti isi komentar, bukan kata tunggal: "ending" bisa berarti
  akhir stream (bukan spoiler) atau ending game (S1–S2).

## Lampiran B — Tabel file hasil

| File | Isi |
|---|---|
| `hasil/pemilihan_video.md` | Tabel 11 video, kriteria, tanggal pengambilan |
| `hasil/kesepakatan.txt` | Kappa 0,740, tabel silang subset 100 |
| `hasil/audit_label_timestamp.txt` | Audit label berbasis waktu pengeditan |
| `hasil/params_metode_a.json` | Parameter terpilih A + log grid validasi |
| `hasil/prediksi_metode_a.csv` | Prediksi uji A (897 baris) |
| `hasil/prediksi_metode_b_seed{42,43,44}.csv` | Prediksi uji B per seed |
| `hasil/metrik.csv` | Metrik kedua metode (B = mean ± std) |
| `hasil/metrik_metode_b_seeds.csv` | Metrik per seed B |
| `hasil/bootstrap_ci.csv` | CI 95% bootstrap F1 spoiler |
| `hasil/mcnemar.txt` | McNemar per seed |
| `hasil/confusion_matrix_*.png` | Confusion matrix (A; B per seed) |
| `hasil/analisis_kesalahan.csv` | 64 baris salah klasifikasi + kolom pola/jenis_galat |
