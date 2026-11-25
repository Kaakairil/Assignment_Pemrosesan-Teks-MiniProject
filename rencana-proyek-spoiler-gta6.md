# Mini Project Pemrosesan Teks: Perbandingan Metode Deteksi Spoiler pada Komentar Replay GTA 6

## 1. Ringkasan

Proyek ini membandingkan dua metode klasifikasi untuk mendeteksi komentar spoiler pada kolom komentar video replay YouTube Windah Basudara yang membahas GTA 6.

- **Metode A (baseline):** TF-IDF + Logistic Regression
- **Metode B (pembanding):** IndoBERT yang di-fine-tune untuk klasifikasi

**Pertanyaan penelitian**
1. Seberapa besar peningkatan F1 dan recall kelas spoiler jika memakai IndoBERT dibanding TF-IDF + Logistic Regression?
2. Pada jenis komentar apa kedua metode paling sering salah?

**Hipotesis**
IndoBERT memberikan F1 kelas spoiler yang lebih tinggi daripada baseline, terutama pada komentar dengan bahasa campuran dan sindiran, karena model transformer memahami konteks kalimat.

**Luaran yang diminta**
- Kode dan dataset (tanpa nama pengguna)
- Panduan label dan hasil kesepakatan antarlabel
- Laporan hasil dan analisis kesalahan

---

## 2. Bahasa dan Lingkungan

| Komponen | Pilihan |
|---|---|
| Bahasa | Python 3.10 atau lebih baru |
| Lingkungan | Google Colab (disarankan untuk IndoBERT, gunakan GPU) atau laptop dengan Jupyter |
| Pengelolaan paket | `pip` atau `conda`, catat versi pada `requirements.txt` |

**Library utama**
- Pengambilan data: `google-api-python-client`
- Pengolahan teks: `pandas`, `re`, `Sastrawi`, `nltk` (opsional)
- Kamus alay: file CSV kamus normalisasi (misalnya dari repositori publik yang lisensinya jelas, atau dibuat sendiri)
- Metode A: `scikit-learn`
- Metode B: `transformers`, `torch`, `datasets`, `accelerate`
- Evaluasi dan visualisasi: `scikit-learn`, `matplotlib`, `seaborn`
- Kesepakatan label: `scikit-learn` (`cohen_kappa_score`)

---

## 3. Struktur Folder

```
proyek-spoiler-gta6/
├── data/
│   ├── raw/              # komentar mentah dari API (JANGAN dibagikan publik)
│   ├── interim/          # hasil pembersihan awal
│   └── labeled/          # data berlabel final (tanpa username)
├── panduan/
│   └── panduan_label.md  # aturan pelabelan
├── notebooks/
│   ├── 01_pengambilan_data.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_kesepakatan_label.ipynb
│   ├── 04_metode_a_tfidf_lr.ipynb
│   ├── 05_metode_b_indobert.ipynb
│   └── 06_evaluasi_analisis.ipynb
├── src/
│   ├── ambil_komentar.py
│   ├── preprocessing.py
│   └── utils.py
├── hasil/
│   ├── metrik.csv
│   └── confusion_matrix/
├── requirements.txt
└── README.md
```

---

## 4. Tahap Pengerjaan

### Tahap 1: Persiapan sumber data

1. **Pilih video replay.** Catat ID video, judul, tanggal unggah, dan durasi. Pastikan video memang replay Windah Basudara tentang GTA 6.
2. **Cek kebijakan.** Baca syarat layanan YouTube dan Google, dan pastikan pengambilan komentar hanya untuk keperluan akademik.
3. **Buat kredensial.** Aktifkan YouTube Data API v3 di Google Cloud Console dan simpan API key di file `.env`, jangan ditulis di kode atau dibagikan.
4. **Verifikasi jumlah ending.** Informasi bahwa GTA 6 memiliki lebih dari 13 ending belum diverifikasi. Cek dari sumber resmi atau wiki game sebelum dicantumkan di laporan.

### Tahap 2: Pengambilan data (minimal 500 data)

**Endpoint:** `commentThreads.list` dengan parameter `videoId`, `part=snippet`, `textFormat=plainText`, `maxResults=100`, dan `pageToken` untuk paginasi.

**Kolom yang disimpan**
| Kolom | Keterangan |
|---|---|
| `id` | ID komentar dari API |
| `teks` | Isi komentar (`textDisplay`) |
| `waktu` | `publishedAt` |
| `jumlah_like` | `likeCount` |
| `balasan` | `totalReplyCount` (opsional) |
| `id_penulis_anonim` | Hash dari ID penulis, bukan ID asli |

**Aturan pengambilan**
- Target awal 1000 komentar mentah, karena sebagian akan terbuang saat pembersihan.
- Simpan respons API mentah sebelum diproses, agar bisa diulang jika ada kesalahan.
- Catat tanggal pengambilan, karena jumlah komentar bisa berubah.
- Komentar balasan dari akun Windah tidak dipakai untuk pelatihan, kecuali diputuskan lain dan dicatat di laporan.

**Penanganan kuota:** API memiliki kuota harian. Catat jumlah permintaan yang dipakai dan lanjutkan keesokan harinya jika perlu.

### Tahap 3: Pembersihan dan preprocessing

Urutan yang disarankan:

1. **Hapus data tidak valid:** komentar kosong, komentar hanya emoji atau tautan, dan duplikat (`drop_duplicates` pada teks yang sudah dinormalisasi).
2. **Hapus informasi pribadi:** nama pengguna, ID penulis asli, dan tautan profil.
3. **Pembersihan teks:**
   - Ubah ke huruf kecil
   - Hapus URL, mention (`@nama`), dan tanda baca berlebih
   - Batasi pengulangan huruf lebih dari dua (`wkwkwkwk` tetap dipertahankan sebagai bentuk ekspresi, catat keputusan ini)
4. **Normalisasi kata gaul:** ganti kata seperti "gk", "ga", "gak", "nggak" dengan kamus normalisasi yang sudah ditetapkan.
5. **Tokenisasi, stopword, dan stemming:** gunakan Sastrawi untuk stopword dan stemming.

**Catatan penting:** Untuk Metode B (IndoBERT), jangan lakukan stemming dan penghapusan stopword. Model transformer dilatih dengan teks alami, jadi pembersihan berlebih bisa merusak konteks. Simpan dua versi teks:
- `teks_klasik`: sudah stemming dan tanpa stopword, dipakai Metode A
- `teks_bert`: hanya dibersihkan ringan, dipakai Metode B

Perbedaan ini harus dijelaskan di laporan sebagai bagian dari perbandingan yang adil.

### Tahap 4: Panduan label

Tulis `panduan/panduan_label.md` sebelum pelabelan dimulai.

**Definisi label**
- **Spoiler (1):** komentar yang mengungkap alur, akhir cerita, nasib karakter, twist, atau peristiwa yang belum tentu diketahui pembaca yang belum menamatkan game.
- **Bukan spoiler (0):** opini, pujian atau kritik tanpa detail cerita, reaksi emosi tanpa menyebut kejadian, pertanyaan, atau peringatan spoiler tanpa isi.
- **Ambigu (-1):** komentar yang tidak bisa diputuskan dengan jelas. Tidak dipakai untuk pelatihan, tetapi dicatat untuk analisis.

**Aturan kasus abu-abu**

| Contoh | Label |
|---|---|
| "Endingnya keren banget" | Bukan spoiler |
| "Endingnya bikin nangis" | Bukan spoiler |
| "Pas karakter X mati aku nangis" | Spoiler |
| "Ending Windah beda sama yang di trailer" | Bukan spoiler |
| "Ending ke-7 yang paling bagus" | Bukan spoiler |
| "Ending ke-7 yang dia dapet itu X" | Spoiler |
| "Jangan spoiler ya, tapi endingnya..." lalu isinya | Spoiler |
| "Jangan spoiler!" saja | Bukan spoiler |
| "Di menit 2:30 ternyata..." lalu menjelaskan | Spoiler |

**Aturan tambahan:** Labeler hanya boleh memakai teks komentar dan panduan. Pengetahuan pribadi tentang ending GTA 6 tidak boleh dipakai. Jika perlu konteks video, gunakan catatan kejadian yang disepakati dan ditulis di panduan.

### Tahap 5: Pelabelan dan kesepakatan antarlabel

1. Dua pelabel (A dan B) melabeli 100 komentar yang sama secara terpisah.
2. Hitung Cohen's kappa.
3. Jika kappa di bawah sekitar 0,6, diskusikan perbedaan, perbaiki panduan, lalu ulangi pada subset lain.
4. Setelah kesepakatan cukup, bagi sisa data dan lanjutkan pelabelan.
5. Catat jumlah tiap label dan jumlah ambigu.

**Target:** kappa minimal sekitar 0,6 dan dokumentasi keputusan dalam `panduan_label.md`.

### Tahap 6: Pembagian data

- Buang komentar ambigu dari data latih dan uji.
- Bagi dengan **stratified split 80:20** (`random_state` tetap, misalnya 42).
- Jika ada validasi, ambil dari data latih, misalnya 10 persen, bukan dari data uji.
- Jika kelas spoiler jauh lebih sedikit, gunakan `class_weight='balanced'` (Metode A) atau `weight` pada loss (Metode B) pada data latih saja.

### Tahap 7: Metode A (baseline) TF-IDF + Logistic Regression

**Pipeline**
- `TfidfVectorizer` dengan `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`
- `LogisticRegression` dengan `class_weight='balanced'`, `max_iter=1000`

**Pencarian parameter:** `GridSearchCV` dengan validasi silang 5-fold pada data latih. Parameter yang diuji: `C`, `ngram_range`, dan `max_features`.

**Catatan:** Simpan model dan vectorizer, lalu jangan melakukan fit ulang pada data uji.

### Tahap 8: Metode B (pembanding) IndoBERT

**Model:** pilih checkpoint IndoBERT dari Hugging Face (misalnya `indobenchmark/indobert-base-p1` atau varian yang sesuai). Cek lisensi dan dokumentasinya.

**Pengaturan fine-tuning**
- `AutoTokenizer` dan `AutoModelForSequenceClassification` dengan `num_labels=2`
- Panjang maksimal token: tentukan dari distribusi panjang komentar (misalnya 128)
- Learning rate: sekitar `2e-5` sampai `3e-5`
- Batch size: 16 atau 32, sesuai memori GPU
- Epoch: 3 sampai 5, dengan early stopping berdasarkan F1 kelas spoiler pada validasi
- Optimizer: AdamW, dengan warmup kecil

**Catatan:** Jalankan setiap eksperimen dengan beberapa seed (misalnya 3 seed) dan laporkan rata-rata dan simpangan bakunya, karena hasil BERT bisa berbeda antar seed.

### Tahap 9: Evaluasi

**Metrik utama**
- Precision, recall, dan F1 untuk kelas spoiler
- Macro F1 sebagai metrik pendukung
- Accuracy hanya sebagai informasi tambahan

**Alasan:** Recall kelas spoiler penting, karena spoiler yang terlewat berdampak lebih besar bagi pembaca.

**Output**
- Tabel perbandingan kedua metode
- Confusion matrix untuk masing-masing metode
- Uji signifikansi sederhana, misalnya McNemar test pada prediksi di data uji yang sama, atau bootstrap untuk selisih F1

### Tahap 10: Analisis kesalahan

1. Kumpulkan semua false negative (spoiler yang tidak terdeteksi) dan false positive (bukan spoiler yang terdeteksi spoiler).
2. Kelompokkan berdasarkan pola:
   - Sindiran atau ironi
   - Bahasa campuran Indonesia–Inggris
   - Komentar sangat pendek
   - Komentar yang hanya menyebut nama karakter
   - Komentar yang menyebut ending tanpa penjelasan
3. Bandingkan apakah pola salah yang sama muncul di kedua metode, atau IndoBERT memperbaiki kelompok tertentu.
4. Sertakan 5 sampai 10 contoh per kelompok (tanpa nama pengguna).

---

## 5. Pembagian Tugas Kelompok (jika dikerjakan berdua atau lebih)

| Tugas | Penanggung jawab |
|---|---|
| Pengambilan data dan keamanan API key | Orang 1 |
| Preprocessing dan kamus normalisasi | Orang 1 |
| Pelabelan dan panduan label | Orang 2 dan 3 |
| Metode A | Orang 2 |
| Metode B | Orang 3 |
| Evaluasi, visualisasi, dan laporan | Bersama |

---

## 6. Jadwal Kerja (contoh 3 minggu)

| Minggu | Kegiatan |
|---|---|
| 1 | Persiapan API, verifikasi video dan ending, pengambilan data, preprocessing awal |
| 2 | Panduan label, pelabelan, kesepakatan kappa, pembagian data |
| 3 | Metode A, Metode B, evaluasi, analisis kesalahan, penulisan laporan |

Sesuaikan dengan tenggat dosen.

---

## 7. Isi Laporan yang Disarankan

1. Pendahuluan: latar belakang, rumusan masalah, dan hipotesis
2. Tinjauan pustaka: deteksi spoiler, TF-IDF, dan IndoBERT (cari sumber akademik yang relevan)
3. Metodologi: sumber data, preprocessing, panduan label, pembagian data, kedua metode
4. Hasil: tabel metrik, confusion matrix, dan uji signifikansi
5. Analisis kesalahan
6. Keterbatasan: data dari satu kanal, satu video, dan label yang subjektif
7. Kesimpulan dan saran
8. Lampiran: panduan label, contoh data (tanpa nama pengguna), dan kode

---

## 8. Etika dan Batasan

- Komentar hanya dipakai untuk penelitian. Jangan menampilkan ulang komentar dengan nama pengguna.
- Hapus identitas pengguna sebelum data dibagikan ke kelompok atau dosen.
- Spoiler bersifat subjektif bergantung pada apakah pembaca sudah menamatkan game. Tetapkan asumsi ini di laporan.
- Data dari satu kanal dan satu video tidak mewakili seluruh penonton GTA 6. Jelaskan keterbatasan ini.
- Jangan mengklaim jumlah ending atau detail cerita tanpa verifikasi dari sumber yang bisa dicek.

---

## 9. Checklist Akhir

- [ ] ID video dan tanggal pengambilan tercatat
- [ ] API key tersimpan aman dan tidak ada di kode
- [ ] Minimal 500 data bersih, tanpa informasi pribadi
- [ ] Panduan label final dan dibagikan ke pelabel
- [ ] Cohen's kappa dilaporkan
- [ ] Dua versi teks (klasik dan BERT) dibuat
- [ ] Metode A dan B dijalankan pada data uji yang sama
- [ ] Metrik untuk kelas spoiler dilaporkan
- [ ] Analisis kesalahan dengan contoh
- [ ] Keterbatasan dan etika ditulis
- [ ] Kode dan README bisa dijalankan ulang
