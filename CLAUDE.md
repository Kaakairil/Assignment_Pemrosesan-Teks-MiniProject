# CLAUDE.md — Proyek Deteksi Spoiler Komentar Replay GTA 6

Dokumen ini adalah instruksi kerja untuk Claude Code di terminal. Baca seluruh dokumen sebelum mengubah kode. Jika ada bagian yang bertentangan dengan permintaan saya di chat, tanyakan dulu sebelum bertindak.

---

## 1. Tujuan proyek

Membandingkan dua metode klasifikasi biner untuk mendeteksi komentar spoiler pada kolom komentar video replay YouTube Windah Basudara tentang GTA 6.

- **Metode A (baseline):** TF-IDF + Logistic Regression
- **Metode B (pembanding):** IndoBERT yang di-fine-tune untuk klasifikasi
- **Label:** `1` = spoiler, `0` = bukan spoiler, `-1` = ambigu (tidak dipakai untuk latih dan uji)

**Pertanyaan penelitian**
1. Seberapa besar peningkatan F1 dan recall kelas spoiler jika memakai IndoBERT dibanding TF-IDF + Logistic Regression?
2. Pada jenis komentar apa kedua metode paling sering salah?

**Hipotesis:** IndoBERT memberi F1 kelas spoiler lebih tinggi, terutama pada komentar bahasa campuran dan sindiran.

---

## 2. Yang TIDAK boleh dilakukan

- JANGAN menulis API key di kode, README, commit, atau log. Key hanya dibaca dari `.env`.
- JANGAN mencetak (`print`) isi `.env` atau key ke terminal.
- JANGAN menyimpan nama pengguna, ID channel penulis, atau tautan profil di file yang masuk ke `data/labeled/` atau `hasil/`.
- JANGAN melakukan fit ulang vectorizer, scaler, atau model pada data uji.
- JANGAN membuat data sintetis atau mengarang komentar untuk menambah jumlah data.
- JANGAN mengklaim jumlah ending GTA 6 atau detail cerita tanpa sumber yang bisa diverifikasi. Tulis `TODO_VERIFIKASI` di laporan jika belum pasti.
- JANGAN menjalankan `git push`, menghapus folder `data/raw/`, atau menimpa file label tanpa konfirmasi saya.
- JANGAN menambahkan dependensi baru tanpa mencatatnya di `requirements.txt`.

---

## 3. Lingkungan dan perintah dasar

Bahasa: **Python 3.10+**.

### Setup awal (jalankan sekali)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### `requirements.txt` (isi minimal)

```
google-api-python-client==2.*
python-dotenv==1.*
pandas==2.*
numpy==1.*
scikit-learn==1.*
Sastrawi==1.*
matplotlib==3.*
seaborn==0.*
torch
transformers==4.*
datasets==2.*
accelerate==0.*
scipy==1.*
statsmodels==0.*
pytest==8.*
tqdm==4.*
```

Versi di atas adalah pola awal. Sebelum commit pertama, jalankan `pip freeze > requirements.lock.txt` dan pakai versi yang benar-benar terpasang.

### `.env` (buat manual, JANGAN di-commit)

```
YOUTUBE_API_KEY=isi_key_saya
VIDEO_ID=isi_id_video_replay
HASH_SALT=isi_string_acak_panjang
```

### `.gitignore` (wajib dibuat sebelum git init)

```
.env
.venv/
data/raw/
__pycache__/
*.pt
*.bin
checkpoints/
.ipynb_checkpoints/
```

Catatan: `data/raw/` tidak di-commit karena berisi data mentah. Yang dibagikan adalah `data/labeled/` yang sudah dianonimkan.

---

## 4. Struktur folder

Buat struktur ini persis seperti berikut:

```
proyek-spoiler-gta6/
├── CLAUDE.md
├── README.md
├── .env                      # tidak di-commit
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/                  # JSON respons API mentah (tidak di-commit)
│   ├── interim/              # komentar hasil ekstraksi dan pembersihan awal
│   └── labeled/              # data berlabel final, tanpa identitas
├── panduan/
│   └── panduan_label.md
├── kamus/
│   └── normalisasi_alay.csv  # kolom: kata_alay,kata_baku
├── src/
│   ├── __init__.py
│   ├── config.py             # konstanta, path, seed
│   ├── ambil_komentar.py     # ekstraksi via YouTube Data API
│   ├── preprocessing.py      # pembersihan, dua versi teks
│   ├── kesepakatan.py        # Cohen's kappa
│   ├── split.py              # pembagian data
│   ├── metode_a.py           # TF-IDF + Logistic Regression
│   ├── metode_b.py           # IndoBERT fine-tuning
│   ├── evaluasi.py           # metrik, confusion matrix, McNemar
│   └── analisis_kesalahan.py # ekspor contoh salah klasifikasi
├── tests/
│   ├── test_preprocessing.py
│   └── test_ambil_komentar.py
├── notebooks/                # opsional, untuk eksplorasi saja
├── hasil/
│   ├── metrik.csv
│   ├── confusion_matrix_metode_a.png
│   ├── confusion_matrix_metode_b.png
│   └── analisis_kesalahan.csv
└── models/                   # checkpoint lokal (tidak di-commit)
```

---

## 5. Konfigurasi global (`src/config.py`)

Semua nilai berikut harus berasal dari satu file ini. Jangan menulis angka ajaib di modul lain.

```python
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1                  # diambil dari data latih
MIN_KAPPA = 0.6
MAX_LEN_BERT = 128
MODEL_BERT = "indobenchmark/indobert-base-p1"   # cek lisensi dan ganti bila perlu
SEEDS_BERT = [42, 43, 44]
MIN_KOMENTAR_BERSIH = 500
TARGET_KOMENTAR_MENTAH = 1000
```

Path diambil relatif terhadap root proyek menggunakan `pathlib.Path(__file__).resolve().parents[1]`.

---

## 6. Skema data

### 6.1 Data mentah (`data/raw/komentar_<VIDEO_ID>_<tanggal>.json`)
Simpan respons API apa adanya, per halaman, agar bisa diulang tanpa memanggil API lagi.

### 6.2 Data interim (`data/interim/komentar_bersih.csv`)

| Kolom | Tipe | Keterangan |
|---|---|---|
| `id` | str | `comment_id` dari API |
| `teks` | str | `textDisplay` asli |
| `waktu` | str | `publishedAt` (ISO 8601) |
| `jumlah_like` | int | `likeCount` |
| `id_anonim` | str | SHA-256 dari (HASH_SALT + author channel id), 16 karakter pertama |
| `teks_klasik` | str | hasil Metode A (stemming, tanpa stopword) |
| `teks_bert` | str | hasil Metode B (pembersihan ringan) |

### 6.3 Data label (`data/labeled/komentar_berlabel.csv`)

| Kolom | Keterangan |
|---|---|
| `id` | sama dengan interim |
| `teks_bert` | untuk pelatihan |
| `label_a` | label pelabel A: 0, 1, atau -1 |
| `label_b` | label pelabel B: 0, 1, atau -1 |
| `label_final` | label setelah diskusi: 0, 1, atau -1 |
| `catatan` | alasan jika ambigu atau diubah |

Kolom `id_anonim`, `waktu`, dan `jumlah_like` boleh ikut, tetapi `id` asli tidak boleh berisi nama pengguna.

---

## 7. Tahap kerja dan spesifikasi per modul

Kerjakan **berurutan**. Setiap tahap punya kriteria selesai. Jangan lanjut ke tahap berikutnya sebelum kriteria tahap sekarang terpenuhi dan saya setujui.

### Tahap 1 — Pengambilan komentar (`src/ambil_komentar.py`)

**Fungsi yang harus ada**
- `muat_key() -> str`: baca `YOUTUBE_API_KEY` dari `.env`. Jika kosong, hentikan program dengan pesan jelas tanpa menampilkan nilai key.
- `ambil_semua_komentar(video_id: str, api_key: str, maks_komentar: int) -> list[dict]`: ambil `commentThreads.list` dengan `part="snippet"`, `textFormat="plainText"`, `maxResults=100`, dan ikuti `nextPageToken` sampai `maks_komentar` tercapai atau halaman habis.
- `simpan_mentah(respons: dict, video_id: str) -> Path`: simpan setiap halaman ke `data/raw/`.
- `ke_dataframe(items: list[dict], salt: str) -> pd.DataFrame`: ubah ke kolom skema 6.2 (tanpa `teks_klasik` dan `teks_bert`), anonimkan author dengan SHA-256.

**Penanganan error yang wajib**
- `HttpError` 403 dengan alasan `commentsDisabled`: catat ke log dan lewati video, jangan crash.
- `HttpError` 403 dengan alasan `quotaExceeded`: simpan progres yang sudah ada, hentikan dengan pesan, dan jangan mencoba ulang berulang kali.
- Batasi jeda antarpanggilan dengan `time.sleep(0.1)` agar tidak terlalu cepat.

**CLI**
```bash
python -m src.ambil_komentar --video-id $VIDEO_ID --maks 1000
```

**Kriteria selesai:** file `data/raw/*.json` ada, `data/interim/komentar_mentah.csv` berisi minimal 800 baris, tidak ada kolom nama pengguna, dan test `tests/test_ambil_komentar.py` lulus dengan data tiruan (mock), bukan panggilan API sungguhan.

### Tahap 2 — Preprocessing (`src/preprocessing.py`)

**Fungsi yang harus ada**
- `bersihkan_dasar(teks: str) -> str`: lowercase, hapus URL, hapus mention `@\w+`, hapus whitespace berlebih, dan hapus karakter kontrol. Emoji dibiarkan di versi BERT, dihapus di versi klasik.
- `normalisasi_alay(teks: str, kamus: dict[str, str]) -> str`: ganti token berdasarkan kamus, token per token dengan pemisahan spasi.
- ` ulang_huruf(teks: str) -> str`: batasi pengulangan huruf lebih dari dua menjadi dua (contoh: "bangettt" menjadi "banget"). Untuk "wkwkwk", jangan dihapus, cukup dinormalisasi menjadi bentuk konsisten "wkwk".
- `teks_klasik(teks: str) -> str`: `bersihkan_dasar` + hapus emoji + `normalisasi_alay` + hapus stopword Sastrawi + stemming Sastrawi.
- `teks_bert(teks: str) -> str`: `bersihkan_dasar` + `normalisasi_alay` + `ulang_huruf`. Tanpa stemming, tanpa penghapusan stopword.
- `hapus_tidak_valid(df: pd.DataFrame) -> pd.DataFrame`: hapus teks kosong, teks yang hanya emoji atau tautan, dan duplikat berdasarkan `teks_bert`. Catat jumlah yang dibuang per kategori ke log.

**Aturan**
- Fungsi harus murni (tanpa efek samping) agar bisa diuji.
- Stemmer dan stopword remover Sastrawi dibuat sekali di level modul, bukan di dalam loop.

**Test wajib (`tests/test_preprocessing.py`)**
- URL dan mention hilang.
- "gk" menjadi "tidak" atau bentuk baku sesuai kamus.
- "bangettt" menjadi "banget".
- `teks_bert` tidak mengubah kata seperti "menangis" menjadi bentuk dasar.
- Komentar kosong dan duplikat dibuang.

**CLI**
```bash
python -m src.preprocessing --input data/interim/komentar_mentah.csv --kamus kamus/normalisasi_alay.csv --output data/interim/komentar_bersih.csv
```

**Kriteria selesai:** semua test lulus, dan file output berisi kolom `teks_klasik` dan `teks_bert`.

### Tahap 3 — Panduan label dan lembar pelabelan

**File `panduan/panduan_label.md`** harus memuat:
1. Definisi spoiler dan bukan spoiler (lihat bagian 8).
2. Contoh tabel kasus abu-abu (lihat bagian 8).
3. Aturan: pelabel hanya boleh memakai teks komentar dan panduan. Pengetahuan pribadi tentang ending tidak boleh dipakai.
4. Daftar catatan kejadian video yang disepakati (dibuat manual oleh saya, bukan oleh Claude).

**File `data/labeled/lembar_label_a.csv` dan `lembar_label_b.csv`** dibuat dari `komentar_bersih.csv`, berisi kolom `id`, `teks`, `label`, `catatan`. Label kosong berarti belum dilabeli.

**Subset kesepakatan:** pilih 100 komentar acak dengan `random_state=SEED` yang sama di kedua lembar. Simpan daftar ID-nya di `data/labeled/subset_kesepakatan.csv`.

**Jangan** membuat label otomatis dan menyebutnya sebagai label manusia. Jika saya meminta bantuan pelabelan awal dengan model, tandai kolom sebagai `label_model` dan jelaskan di laporan.

### Tahap 4 — Kesepakatan antarlabel (`src/kesepakatan.py`)

- Baca dua lembar label untuk subset 100 data.
- Hitung `cohen_kappa_score` dengan label `-1` dikeluarkan dari perhitungan.
- Cetak kappa, jumlah pasangan yang dipakai, dan tabel silang (`pd.crosstab`).
- Simpan hasil ke `hasil/kesepakatan.txt`.

**Aturan:** jika kappa < `MIN_KAPPA`, hentikan proses dan minta saya memperbaiki panduan. Jangan lanjut ke pelabelan penuh.

**Kriteria selesai:** kappa dilaporkan, dan tahap pelabelan penuh hanya dilanjutkan jika kappa memenuhi ambang.

### Tahap 5 — Gabung label dan pembagian data (`src/split.py`)

- Gabungkan `label_final` ke `komentar_bersih.csv` menjadi `data/labeled/komentar_berlabel.csv`.
- Buang baris dengan `label_final == -1` dari data latih dan uji. Simpan jumlah yang dibuang di log.
- Pembagian: `train_test_split` dengan `test_size=TEST_SIZE`, `stratify=label`, `random_state=SEED`.
- Dari data latih, ambil `VAL_SIZE` untuk validasi dengan stratifikasi juga.
- Simpan indeks atau file terpisah: `data/labeled/train.csv`, `val.csv`, `test.csv`.

**Aturan:** data uji tidak boleh dipakai untuk memilih parameter apa pun. Tulis ini sebagai komentar di kode.

**Kriteria selesai:** tiga file split ada, proporsi kelas di tiap split tercetak dan mirip, dan tidak ada ID yang muncul di dua split berbeda (cek dengan assert).

### Tahap 6 — Metode A: TF-IDF + Logistic Regression (`src/metode_a.py`)

**Pipeline**
```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
])
```

**Pencarian parameter**
- `GridSearchCV` dengan `cv=StratifiedKFold(5, shuffle=True, random_state=SEED)` pada data latih saja.
- Grid: `tfidf__ngram_range` in [(1,1), (1,2)], `tfidf__max_features` in [None, 10000], `clf__C` in [0.1, 1, 10].
- Scoring: `f1` untuk kelas 1 (`make_scorer(f1_score, pos_label=1)`).

**Output**
- Simpan model terbaik ke `models/metode_a.joblib`.
- Simpan prediksi pada data uji ke `hasil/prediksi_metode_a.csv` (kolom: `id`, `teks_klasik`, `label_benar`, `prediksi`, `skor`).
- Simpan parameter terpilih ke `hasil/params_metode_a.json`.

**Kriteria selesai:** model tersimpan, prediksi uji tersimpan, dan tidak ada fit pada data uji.

### Tahap 7 — Metode B: IndoBERT (`src/metode_b.py`)

**Persiapan**
- Muat tokenizer dan model dari `MODEL_BERT` dengan `num_labels=2`.
- Tokenisasi `teks_bert` dengan `truncation=True`, `max_length=MAX_LEN_BERT`, dan `padding="max_length"`.
- Jika proporsi token terpotong tinggi (lebih dari 5 persen), laporkan di log dan tanyakan kepada saya sebelum mengubah panjang.

**Pelatihan**
- Gunakan `transformers.Trainer` atau loop PyTorch sederhana. Pilih satu dan konsisten.
- Learning rate `2e-5`, batch size 16 atau 32 (sesuai memori), epoch maksimal 4.
- `evaluation_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="f1"`.
- Bobot kelas: `class_weight` dihitung dari data latih dan dipakai di loss. Jangan menghitung dari data uji.
- Jalankan untuk setiap seed di `SEEDS_BERT`. Simpan checkpoint terbaik per seed di `models/bert_seed<N>/`.

**Output**
- `hasil/prediksi_metode_b_seed<N>.csv` dengan kolom yang sama seperti Metode A.
- `hasil/metrik_metode_b_seeds.csv` berisi rata-rata dan simpangan baku F1 kelas spoiler.

**Batas sumber daya:** jika GPU tidak tersedia, hentikan dan beri tahu saya. Jangan mengganti model ke versi yang lebih kecil tanpa izin.

**Kriteria selesai:** tiga seed selesai dijalankan, prediksi uji tersimpan untuk setiap seed, dan rata-rata serta simpangan baku tercatat.

### Tahap 8 — Evaluasi (`src/evaluasi.py`)

**Metrik untuk kelas 1 (spoiler)**
- Precision, recall, F1.
- Macro F1 sebagai pendukung.
- Accuracy hanya ditampilkan sebagai informasi tambahan.

**Output**
- `hasil/metrik.csv`: baris untuk Metode A, dan baris untuk Metode B (rata-rata ± simpangan baku dari seed).
- `hasil/confusion_matrix_metode_a.png` dan `hasil/confusion_matrix_metode_b.png` dengan judul dan label kelas yang jelas.
- Uji McNemar pada prediksi uji yang sama dari kedua metode. Gunakan `statsmodels.stats.contingency_tables.mcnemar`. Simpan statistik dan p-value ke `hasil/mcnemar.txt`.

**Kriteria selesai:** semua file hasil ada dan angka di `metrik.csv` konsisten dengan prediksi mentah.

### Tahap 9 — Analisis kesalahan (`src/analisis_kesalahan.py`)

- Gabungkan prediksi kedua metode berdasarkan `id`.
- Tandai false negative dan false positive untuk masing-masing metode.
- Tambahkan kolom `pola` yang diisi manual oleh saya (atau kategori dari daftar di bawah): `sindiran`, `bahasa_campuran`, `teks_pendek`, `hanya_nama_karakter`, `menyebut_ending_tanpa_detail`, `lainnya`.
- Simpan ke `hasil/analisis_kesalahan.csv`.
- Cetak 5 sampai 10 contoh per pola tanpa identitas.

**Aturan:** kolom `pola` tidak boleh diisi otomatis tanpa dicek saya, karena ini bagian dari temuan penelitian.

---

## 8. Panduan label (ringkasan untuk `panduan/panduan_label.md`)

**Spoiler (1):** mengungkap alur, akhir cerita, nasib karakter, twist, atau peristiwa penting yang belum diketahui pembaca yang belum menamatkan game. Berlaku juga untuk komentar yang merujuk pada momen atau keputusan spesifik di video replay.

**Bukan spoiler (0):** opini, pujian atau kritik tanpa detail cerita, reaksi emosi tanpa menyebut kejadian, pertanyaan, atau peringatan spoiler tanpa isi.

**Ambigu (-1):** tidak bisa diputuskan dengan jelas. Tidak dipakai untuk latih dan uji.

| Contoh | Label |
|---|---|
| "Endingnya keren banget" | 0 |
| "Endingnya bikin nangis" | 0 |
| "Pas karakter X mati aku nangis" | 1 |
| "Ending Windah beda sama yang di trailer" | 0 |
| "Ending ke-7 yang paling bagus" | 0 |
| "Ending ke-7 yang dia dapet itu X" | 1 |
| "Jangan spoiler ya, tapi endingnya..." lalu isinya | 1 |
| "Jangan spoiler!" saja | 0 |
| "Di menit 2:30 ternyata..." lalu menjelaskan | 1 |

---

## 9. Pengujian dan kualitas kode

- Jalankan `pytest -q` sebelum setiap commit.
- Setiap fungsi di `src/` yang punya logika (bukan hanya pemanggilan library) harus punya minimal satu test.
- Gunakan type hint dan docstring singkat dalam bahasa Indonesia.
- Setiap skrip punya `if __name__ == "__main__":` dan `argparse`.
- Gunakan logging (`logging` standar), bukan `print`, untuk pesan proses.
- Tetapkan seed di setiap tempat yang memakai acak (`random`, `numpy`, `torch`).

---

## 10. Git (opsional, jika dipakai)

```bash
git init
git add .gitignore CLAUDE.md README.md requirements.txt src tests panduan kamus
git commit -m "Setup awal proyek"
```

Jangan `git add` folder `data/raw/`, `models/`, atau `.env`. Periksa dengan `git status` sebelum commit.

---

## 11. Cara Claude Code bekerja dengan proyek ini

1. Baca `CLAUDE.md` ini dan `README.md` (jika ada) sebelum mengubah apa pun.
2. Kerjakan satu tahap pada satu waktu. Di akhir tahap, jalankan test dan tampilkan ringkasan: file yang dibuat, jumlah data, dan hasil test.
3. Jika menemukan masalah yang tidak tercakup di dokumen ini (misalnya API menolak video, atau kamus tidak tersedia), berhenti dan tanyakan.
4. Jangan memperluas cakupan proyek (misalnya menambah metode ketiga atau kanal lain) kecuali saya memintanya.
5. Setiap keputusan yang memengaruhi hasil (ambang kappa, panjang token, pilihan kamus, pembuangan data) harus dicatat di `README.md` bagian "Keputusan".

---

## 12. Placeholder yang harus diisi saya

- `VIDEO_ID` di `.env`: ID video replay Windah Basudara yang dipilih.
- `HASH_SALT` di `.env`: string acak panjang.
- Jumlah ending GTA 6: `TODO_VERIFIKASI` sampai ada sumber resmi.
- Panduan catatan kejadian video: dibuat manual oleh saya.
- Checkpoint IndoBERT: pastikan `MODEL_BERT` sesuai dengan dokumentasi dan lisensinya.

---

## 13. Checklist sebelum laporan

- [ ] `.env` ada dan tidak di-commit
- [ ] Minimal 500 komentar bersih tanpa identitas
- [ ] Panduan label final dan kappa ≥ 0,6 dilaporkan
- [ ] Split stratified dengan seed tetap, tanpa kebocoran ke data uji
- [ ] Metode A dan B dijalankan pada data uji yang sama
- [ ] Tiga seed untuk Metode B, dengan rata-rata dan simpangan baku
- [ ] Metrik kelas spoiler, confusion matrix, dan McNemar tersimpan
- [ ] Analisis kesalahan dengan contoh tanpa identitas
- [ ] `pytest` lulus
- [ ] Keterbatasan dan etika ditulis di laporan
