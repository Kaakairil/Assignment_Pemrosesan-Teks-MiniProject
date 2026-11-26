# CLAUDE.md — Deteksi Spoiler Komentar Video GTA 6 (Windah Basudara)

Instruksi kerja untuk Claude Code di terminal. Baca seluruh dokumen ini sebelum mengubah kode. Jika ada yang bertentangan dengan permintaan saya di chat, tanyakan dulu sebelum bertindak.

---

## 1. Tujuan

Membandingkan dua metode klasifikasi biner untuk mendeteksi komentar spoiler pada kolom komentar video arsip Windah Basudara, seri "GTA 6 Gameplay" #1 sampai #11.

- **Metode A (baseline):** TF-IDF + Logistic Regression
- **Metode B (pembanding):** IndoBERT yang di-fine-tune
- **Label:** `1` = spoiler, `0` = bukan spoiler, `-1` = ambigu (tidak dipakai untuk latih, validasi, atau uji)

**Pertanyaan penelitian**
1. Seberapa besar peningkatan F1 dan recall kelas spoiler jika memakai IndoBERT dibanding TF-IDF + Logistic Regression?
2. Pada jenis komentar apa kedua metode paling sering salah, dan apakah kesalahan itu berasal dari model atau dari label?

**Hipotesis:** IndoBERT memberi F1 kelas spoiler lebih tinggi, terutama pada komentar bahasa campuran dan sindiran.

---

## 2. Pembagian video (FINAL)

Pembagian dilakukan **per video, berdasarkan urutan episode**. Tidak ada video yang muncul di dua peran.

| Peran | Episode | Video ID | Judul (ringkas) |
|---|---|---|---|
| Latih | #1 | `45r3OyaxvFE` | GTA VI siap tamat 100% |
| Latih | #2 | `QSPPyA1736U` | Jason kangen dan pengen ketemu Lucia |
| Latih | #3 | `B2Bon2X1jPQ` | Ketemu Bigfoot dan ke tempat sepatu |
| Latih | #4 | `03qhcshfeMY` | Jason dan Lucia difitnah |
| Latih | #5 | `lF_QYSm2T-Q` | Ketemu bayi alien dan UFO |
| Latih | #6 | `u_Lb6sh9N0M` | Ganti mobil dan menemukan topi Arthur |
| Latih | #7 | `MWQtXTTAgeM` | Bertemu Red Hulk |
| Latih | #8 | `z6sBvvg_004` | Akhirnya pergi ke Vice City |
| Validasi | #9 | `zYTPbx83wCQ` | Misteri ritual satanis |
| Validasi | #10 | `wJVeWiqMJ-M` | Buka portal rahasia dan baju Tommy Vercetti |
| Uji | #11 | `5m-Xi8lgvWY` | Akhirnya misi epic dan rampok bank |

Aturan:
- **Latih:** episode #1 sampai #8.
- **Validasi:** episode #9 dan #10. Dipakai untuk memilih hyperparameter.
- **Uji:** episode #11 saja. Dipakai **satu kali**, setelah model dan hyperparameter dikunci.
- ID diambil dari parameter `v=` pada URL. Parameter seperti `&t=` dan `&pp=` diabaikan.
- Video di luar tabel ini tidak boleh masuk dataset final.

Konfigurasi di `.env`:

```
YOUTUBE_API_KEY=isi_key_saya
HASH_SALT=isi_string_acak_panjang
TRAIN_VIDEO_IDS=45r3OyaxvFE,QSPPyA1736U,B2Bon2X1jPQ,03qhcshfeMY,lF_QYSm2T-Q,u_Lb6sh9N0M,MWQtXTTAgeM,z6sBvvg_004
VAL_VIDEO_IDS=zYTPbx83wCQ,wJVeWiqMJ-M
TEST_VIDEO_IDS=5m-Xi8lgvWY
```

---

## 2A. Keputusan yang harus ditetapkan sebelum pelabelan

Dua keputusan ini belum final. Keduanya harus dicatat di `README.md` bagian "Keputusan" sebelum ada label manusia.

**A. Sumber data**
- Opsi 1 (sesuai rancangan awal): komentar YouTube dari episode #1 sampai #11, lewat YouTube Data API.
- Opsi 2: komentar Instagram yang dikumpulkan manual. Ini tidak bisa diambil lewat API, dan pengambilan otomatis melanggar syarat layanan Instagram. Jika dipakai, wajib ada kolom `sumber` (`youtube` atau `instagram`), tidak ada identitas akun, dan pembagian per video tidak berlaku untuk sumber ini. Evaluasi harus dilaporkan per sumber.
- Jika keduanya dipakai, jangan menggabungkan data tanpa kolom `sumber`.

**B. Definisi spoiler versi 2.** Lihat bagian 10. Versi 1 tidak lagi dipakai untuk pelabelan. Setiap label versi 1 dianggap tidak valid dan harus dilabeli ulang.

Jika kedua keputusan sudah ditetapkan, catat tanggal dan alasan di README. Setelah itu tidak boleh diubah tanpa mencatat ulang dan melabeli ulang data terdampak.

## 3. Data yang TIDAK boleh masuk pipeline

Berikut sumber yang dilarang untuk latih, validasi, uji, atau laporan hasil. Claude Code harus memeriksa ini secara eksplisit.

1. **Data lama (2.795 komentar dari 10 video awal).** Label-nya heuristik. Data ini hanya boleh dipindahkan ke `data/legacy/` dan tidak dibaca oleh modul split atau evaluasi. Hasil evaluasi dari data ini dianggap preliminer dan tidak dilaporkan sebagai hasil akhir.
2. **Label dari `src/pelabel_awal.py` (kolom `label_model`).** Modul ini dihapus dari pipeline. Jangan memakai kolomnya untuk pelatihan maupun evaluasi.
3. **`data/spoiler_example/`.** Berisi 40 baris contoh dari sumber Instagram, YouTube Shorts, dan artikel, bukan komentar video #1 sampai #11. Isinya juga memuat klaim cerita yang belum diverifikasi. Folder ini hanya untuk latihan pelabelan dan tidak boleh dibaca oleh modul apa pun di `src/`.
4. **`contoh_200_spoiler.csv` (200 baris, 100 `spoiler` dan 100 `non_spoiler`).** Ini bukan komentar YouTube dari video bagian 2, dan labelnya **tidak berasal dari pelabel manusia**. Penggunaan yang diizinkan hanya:
   - menguji kode di `tests/` (misalnya memastikan `split.py` menolak baris yang tidak punya `video_id` valid, dan bahwa preprocessing berjalan pada teks dengan format ini),
   - latihan pelabelan dan tes pemahaman pelabel.
   File ini **tidak boleh** masuk `train.csv`, `val.csv`, `test.csv`, maupun `komentar_berlabel.csv`, dan tidak boleh dicampur ke hasil evaluasi.
   Label dalam file ini juga **bertentangan dengan panduan**: baris dari tipe "Bocoran Mekanik & Misteri" dan "Bocoran Misi & Gameplay Spesifik" berlabel `spoiler`, padahal aturan G1 menyatakan tips dan mekanik bukan spoiler. Jangan memperbaiki label itu secara diam-diam. Catat sebagai temuan di README.
5. **Komentar dari video di luar tabel bagian 2.**
5. **Teks yang bukan komentar penonton**, misalnya poin ringkasan, judul, atau tautan berulang.
6. **Label yang dibuat Claude atau model lain.** Semua label latih, validasi, dan uji harus berasal dari pelabel manusia.

Catatan teknis: ID pada `spoiler_example` adalah bilangan bulat 1 sampai 40. Jika suatu saat data itu tercampur, ID bisa bertabrakan dengan ID lain, jadi `split.py` harus menolak baris yang tidak punya `video_id` dari tabel bagian 2.

---

## 4. Yang TIDAK boleh dilakukan

- JANGAN menulis API key di kode, README, commit, atau log. Key hanya dibaca dari `.env`.
- JANGAN mencetak isi `.env` atau nilai key ke terminal.
- JANGAN menyimpan nama pengguna, ID channel penulis, atau tautan profil di file yang masuk `data/labeled/` atau `hasil/`.
- JANGAN memakai `StratifiedKFold` acak untuk memilih hyperparameter, karena komentar dari video yang sama akan bocor antar lipatan.
- JANGAN melakukan fit ulang vectorizer, scaler, atau model pada data uji.
- JANGAN memilih seed atau model berdasarkan hasil pada data uji.
- JANGAN membuat data sintetis atau mengarang komentar.
- JANGAN mengklaim jumlah ending GTA 6 atau detail cerita tanpa sumber. Tulis `TODO_VERIFIKASI`.
- JANGAN membuat kode untuk mengambil komentar Instagram secara otomatis, baik dengan scraping maupun login.
- JANGAN memakai label yang dibuat dengan panduan versi 1.0 untuk pelatihan, validasi, atau uji.
- JANGAN menjalankan `git push`, menghapus `data/raw/`, atau menimpa file label tanpa konfirmasi saya.
- JANGAN menambah dependensi tanpa mencatatnya di `requirements.txt`.

---

## 5. Lingkungan

Bahasa: **Python 3.10+**.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock.txt   # setelah instalasi pertama
```

`requirements.txt` berisi: `google-api-python-client`, `python-dotenv`, `pandas`, `numpy`, `scikit-learn`, `Sastrawi`, `matplotlib`, `seaborn`, `torch`, `transformers`, `datasets`, `accelerate`, `scipy`, `statsmodels`, `pytest`, `tqdm`. Pakai versi yang benar-benar terpasang, dan catat di `requirements.lock.txt`.

`.gitignore` wajib dibuat sebelum `git init`:

```
.env
.venv/
data/raw/
data/legacy/
data/spoiler_example/
models/
__pycache__/
*.pt
*.bin
.ipynb_checkpoints/
```

---

## 6. Struktur folder

```
proyek-spoiler-gta6/
├── CLAUDE.md
├── README.md                    # termasuk bagian "Keputusan"
├── .env                         # tidak di-commit
├── .gitignore
├── requirements.txt
├── requirements.lock.txt
├── data/
│   ├── raw/                     # JSON respons API per halaman (tidak di-commit)
│   ├── interim/                 # komentar hasil ekstraksi dan preprocessing
│   ├── labeled/                 # data berlabel final, tanpa identitas
│   │   ├── lembar_label_a.csv
│   │   ├── lembar_label_b.csv
│   │   ├── subset_kesepakatan.csv
│   │   ├── komentar_berlabel.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── legacy/                  # data lama, tidak dipakai (lihat bagian 3)
│   └── spoiler_example/         # contoh pelabelan, tidak dipakai pipeline
│       ├── contoh_200_spoiler.csv   # 200 contoh buatan (100 spoiler, 100 non_spoiler), hanya untuk tes kode dan latihan pelabel
│       └── (40 baris contoh sebelumnya, juga tidak dipakai pipeline)
├── panduan/
│   ├── panduan_label.md         # aturan label final, termasuk Lampiran A
│   └── catatan_kejadian.md      # catatan manual per episode, dibuat manusia
├── kamus/
│   └── normalisasi_alay.csv     # kolom: kata_alay,kata_baku
├── src/
│   ├── config.py
│   ├── ambil_komentar.py
│   ├── preprocessing.py
│   ├── kesepakatan.py
│   ├── split.py
│   ├── metode_a.py
│   ├── metode_b.py
│   ├── evaluasi.py
│   └── analisis_kesalahan.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ambil_komentar.py
│   └── test_split.py
├── hasil/
│   ├── pemilihan_video.md
│   ├── kesepakatan.txt
│   ├── metrik.csv
│   ├── bootstrap_ci.csv
│   ├── mcnemar.txt
│   ├── confusion_matrix_metode_a.png
│   ├── confusion_matrix_metode_b.png
│   ├── prediksi_metode_a.csv
│   ├── prediksi_metode_b_seed*.csv
│   └── analisis_kesalahan.csv
└── models/                      # checkpoint lokal (tidak di-commit)
```

---

## 7. Konfigurasi (`src/config.py`)

Semua nilai berasal dari file ini. Jangan menulis angka tetap di modul lain.

```python
SEED = 42
SEEDS_BERT = [42, 43, 44]
MIN_KAPPA = 0.6
MAX_LEN_BERT = 128
MODEL_BERT = "indobenchmark/indobert-base-p1"   # cek lisensi dan ketersediaan sebelum dipakai
N_BOOTSTRAP = 1000
ALPHA_CI = 0.05
TRAIN_VIDEO_IDS = [...]   # dibaca dari .env, lihat bagian 2
VAL_VIDEO_IDS = [...]
TEST_VIDEO_IDS = [...]
```

Path dihitung relatif terhadap root proyek dengan `pathlib.Path(__file__).resolve().parents[1]`. Daftar video di atas harus dibaca dari `.env`, lalu divalidasi bahwa ketiganya tidak beririsan.

---

## 8. Skema data

**Mentah** (`data/raw/komentar_<VIDEO_ID>_<tanggal>.json`): respons API per halaman, tanpa diubah, agar bisa diulang tanpa memanggil API lagi.

**Interim** (`data/interim/komentar_bersih.csv`):

| Kolom | Keterangan |
|---|---|
| `video_id` | ID video asal. Wajib ada. |
| `id` | `comment_id` dari API, dengan awalan `yt_` agar tidak tercampur dengan ID lain |
| `teks` | `textDisplay` asli |
| `waktu` | `publishedAt` (ISO 8601) |
| `jumlah_like` | `likeCount` |
| `id_anonim` | 16 karakter pertama SHA-256 dari (`HASH_SALT` + ID channel penulis) |
| `teks_klasik` | keluaran Metode A |
| `teks_bert` | keluaran Metode B |

**Berlabel** (`data/labeled/komentar_berlabel.csv`): kolom di atas, ditambah `label_a`, `label_b`, `label_final` (0, 1, atau -1), dan `catatan`. Tidak ada kolom `label_model`.

---

## 9. Tahap kerja

Kerjakan berurutan. Jangan lanjut ke tahap berikutnya sebelum kriteria selesai terpenuhi dan saya menyetujui.

### Tahap 1: Pengambilan komentar (`src/ambil_komentar.py`)

- `muat_key() -> str`: baca `YOUTUBE_API_KEY` dari `.env`. Jika kosong, hentikan dengan pesan tanpa menampilkan nilai key.
- `ambil_semua_komentar(video_id, api_key, maks_komentar) -> list[dict]`: `commentThreads.list` dengan `part="snippet"`, `textFormat="plainText"`, `maxResults=100`, ikuti `nextPageToken`. Kuota: 1 unit per panggilan.
- `simpan_mentah(respons, video_id, tanggal) -> Path`: simpan setiap halaman ke `data/raw/`.
- `ke_dataframe(items, salt, video_id) -> DataFrame`: ubah ke skema bagian 8, anonimkan author.
- Error yang wajib ditangani: `commentsDisabled` (catat, lewati video), `quotaExceeded` (simpan progres, hentikan, jangan retry berulang).
- Beri `time.sleep(0.1)` antarpanggilan.

CLI:
```bash
python -m src.ambil_komentar --video-ids $TRAIN_VIDEO_IDS,$VAL_VIDEO_IDS,$TEST_VIDEO_IDS --maks 1000
```

Selesai jika: file mentah per video ada, tanggal pengambilan tercatat, tidak ada kolom identitas, dan `tests/test_ambil_komentar.py` lulus dengan mock.

### Tahap 2: Preprocessing (`src/preprocessing.py`)

- `bersihkan_dasar(teks)`: lowercase, hapus URL dan mention, rapikan spasi dan karakter kontrol.
- `ulang_huruf(teks)`: batasi pengulangan huruf menjadi dua. `wkwkwk` dinormalisasi menjadi `wkwk`.
- `normalisasi_alay(teks, kamus)`: ganti token berdasarkan kamus.
- `teks_klasik(teks)`: `bersihkan_dasar` + hapus emoji + `normalisasi_alay` + stopword Sastrawi + stemming Sastrawi.
- `teks_bert(teks)`: `bersihkan_dasar` + `normalisasi_alay` + `ulang_huruf`. Tanpa stemming dan tanpa penghapusan stopword.
- `hapus_tidak_valid(df)`: hapus kosong, hanya emoji atau tautan, dan duplikat berdasarkan `teks_bert`. Catat jumlah per kategori.

Fungsi harus murni. Stemmer dan stopword dibuat sekali di level modul.

Test wajib: URL dan mention hilang; "gk" dinormalisasi sesuai kamus; "bangettt" menjadi "banget"; `teks_bert` tidak mengubah bentuk kata; komentar kosong dan duplikat dibuang.

Selesai jika: test lulus dan output memuat `teks_klasik` dan `teks_bert`.

### Tahap 3: Panduan label dan pelabelan

- `panduan/panduan_label.md` berisi definisi, aturan S, G, T, R, kasus abu-abu, dan aturan -1, sesuai bagian 10.
- `panduan/catatan_kejadian.md` dibuat manual oleh saya untuk setiap episode #1 sampai #11: nama misi dan peristiwa penting. Claude tidak boleh mengisinya.
- `data/spoiler_example/` hanya boleh dipakai untuk tes pemahaman pelabel (Lampiran B panduan), tidak untuk data.
- Lembar pelabelan `lembar_label_a.csv` dan `lembar_label_b.csv` berisi `video_id`, `id`, `teks`, `label`, `catatan`. Kolom `label` dikosongkan. Hanya nilai 0, 1, atau -1 yang diterima.
- Subset kesepakatan: 100 komentar acak dengan `SEED`, disimpan di `subset_kesepakatan.csv`.
- Jika memakai sampling terarah (misalnya komentar yang memuat kata kunci misi), catat strateginya dan pisahkan dari sampel acak.

Tahap 3 tidak boleh memuat label otomatis. Jika butuh bantuan urutan, boleh mengurutkan komentar berdasarkan kata kunci, asalkan kolom label tetap kosong.

### Tahap 4: Kesepakatan antarlabel (`src/kesepakatan.py`)

- Baca dua lembar label untuk subset 100 komentar.
- Hitung `cohen_kappa_score`. Buang label -1 dari perhitungan.
- Simpan kappa, jumlah pasangan, dan tabel silang ke `hasil/kesepakatan.txt`.
- Jika kappa < `MIN_KAPPA`, hentikan. Perbaiki panduan, lalu ulangi pada subset baru.

Selesai jika: kappa dilaporkan dan kappa memenuhi ambang sebelum pelabelan penuh.

### Tahap 5: Pelabelan penuh dan split (`src/split.py`)

1. Setelah kappa memenuhi ambang, selesaikan pelabelan sisa data. Jika dua pelabel berbeda, putuskan label final dan tulis di `catatan`.
2. Gabungkan ke `komentar_berlabel.csv`.
3. Buang baris `label_final == -1` dari latih, validasi, dan uji. Catat jumlahnya per video.
4. Pembagian per `video_id`, bukan acak:
   - `train.csv`: `video_id` di `TRAIN_VIDEO_IDS`
   - `val.csv`: `video_id` di `VAL_VIDEO_IDS`
   - `test.csv`: `video_id` di `TEST_VIDEO_IDS`
5. Tolak baris yang `video_id`-nya tidak termasuk ketiga daftar.
6. Assert ketiga daftar tidak beririsan dan tidak ada `id` yang muncul di dua split.
7. Cetak jumlah dan proporsi kelas spoiler per split dan per video. Jika spoiler di `test.csv` kurang dari 20, tulis sebagai keterbatasan utama di README.

`tests/test_split.py` wajib memuat: tes bahwa baris dari `spoiler_example` (termasuk `contoh_200_spoiler.csv`) atau legacy ditolak, tes bahwa video di luar tabel ditolak, dan tes bahwa tidak ada irisan.

Catatan label: sebelum data apa pun dipakai, `split.py` harus memeriksa bahwa setiap label `spoiler` punya `video_id` dari tabel bagian 2 dan berasal dari `lembar_label_*.csv` yang sudah disepakati. Label yang tidak punya asal manusia dianggap tidak valid.

Selesai jika: tiga file split ada, assert lulus, dan tes split lulus.

### Tahap 6: Metode A (`src/metode_a.py`)

Pipeline:
```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
])
```

- Hyperparameter dipilih dengan melatih pada `train.csv` dan mengevaluasi pada `val.csv`. Gunakan `PredefinedSplit` dari scikit-learn, atau loop manual. Grid: `ngram_range` in [(1,1), (1,2)], `max_features` in [None, 10000], `C` in [0.1, 1, 10]. Scoring: F1 kelas 1.
- Setelah hyperparameter terkunci, latih ulang pada `train.csv`, lalu evaluasi sekali pada `test.csv`.
- Simpan model ke `models/metode_a.joblib`, prediksi ke `hasil/prediksi_metode_a.csv` (`video_id`, `id`, `teks_klasik`, `label_benar`, `prediksi`, `skor`), dan parameter ke `hasil/params_metode_a.json`.

### Tahap 7: Metode B (`src/metode_b.py`)

- `AutoTokenizer` dan `AutoModelForSequenceClassification` dengan `num_labels=2` dari `MODEL_BERT`.
- Tokenisasi `teks_bert`, `max_length=MAX_LEN_BERT`. Jika lebih dari 5 persen token terpotong, laporkan dan tanyakan sebelum mengubah panjang.
- Learning rate `2e-5`, batch 16 atau 32, maksimal 4 epoch, AdamW dengan warmup kecil.
- Early stopping berdasarkan F1 kelas 1 pada `val.csv`.
- Bobot kelas dihitung dari `train.csv` dan dipakai di loss.
- Jalankan untuk setiap seed di `SEEDS_BERT`. Simpan checkpoint terbaik di `models/bert_seed<N>/`.
- Hasil: `hasil/prediksi_metode_b_seed<N>.csv`, dan `hasil/metrik_metode_b_seeds.csv` berisi rata-rata dan simpangan baku F1 kelas 1.
- Jika GPU tidak tersedia, hentikan dan beri tahu saya. Jangan mengganti model ke versi lebih kecil tanpa izin.

### Tahap 8: Evaluasi (`src/evaluasi.py`)

- Metrik untuk kelas 1: precision, recall, F1. Macro F1 sebagai pendukung. Accuracy hanya informasi tambahan.
- Metode B dilaporkan sebagai rata-rata ± simpangan baku dari tiga seed. **Jangan memilih "konsensus seed" atau seed terbaik.**
- CI 95 persen untuk F1 kelas 1 dengan bootstrap pada `test.csv` (`N_BOOTSTRAP` resampel, berdasarkan komentar).
- McNemar pada prediksi uji yang sama dari A dan B, memakai `statsmodels`. Simpan ke `hasil/mcnemar.txt`.
- Confusion matrix untuk masing-masing metode.
- Jika selisih F1 lebih kecil dari simpangan baku seed atau tumpang tindih dengan CI, tulis bahwa perbedaan tidak terdeteksi.

Selesai jika: semua file hasil ada dan angka di `metrik.csv` konsisten dengan prediksi mentah.

### Tahap 9: Analisis kesalahan (`src/analisis_kesalahan.py`)

- Gabungkan prediksi A dan B berdasarkan `id`.
- Tandai false negative dan false positive per metode.
- Kolom `pola` diisi manual oleh saya dengan kategori: `sindiran`, `bahasa_campuran`, `teks_pendek`, `hanya_nama_karakter`, `menyebut_ending_tanpa_detail`, `tips_mekanik`, `lainnya`.
- Pisahkan kasus **galat label** (label salah menurut panduan) dari **galat model**. Tandai galat label di kolom `jenis_galat`, dan jangan menghitungnya sebagai kegagalan model.
- Cetak 5 sampai 10 contoh per pola tanpa identitas.

Kolom `pola` dan `jenis_galat` tidak boleh diisi otomatis tanpa dicek saya.

---

## 10. Panduan label (ringkas, sumber lengkap di `panduan/panduan_label.md`)

**Versi panduan: 2.1** (menggantikan 1.0 dan 2.0; tanggal dan alasan dicatat di README). Versi 2.1 menambahkan S6 dan menegaskan bahwa jumlah ending tidak menjadi dasar label.

**Spoiler (1):** mengungkap isi cerita atau konsekuensi cerita yang belum diketahui pembaca yang belum menamatkan game: alur, akhir, nasib karakter, twist, peristiwa penting, dan mekanik atau sistem permainan yang menentukan salah satunya (lihat S5). Juga berlaku untuk komentar yang menyebut momen spesifik di video.

**Bukan spoiler (0):** opini, pujian atau kritik tanpa detail cerita, reaksi emosi, harapan, pertanyaan, peringatan spoiler tanpa isi, dan tips atau mekanik yang tidak memengaruhi cerita (lihat G1).

**Aturan mekanik (berlaku mulai versi 2.0):**
- **S5 (mekanik yang memengaruhi cerita):** tips, mekanik, atau sistem permainan dianggap spoiler jika menyebut konsekuensinya terhadap alur, ending, nasib karakter, hubungan antar karakter, atau akses ke misi atau ending tersembunyi. Contoh: "kalau trust meter di bawah 30 persen, Lucia ninggalin Jason" (1).
- **S6 (keputusan yang menentukan ending):** komentar yang menyebut pilihan dialog, misi, atau tindakan tertentu yang menentukan ending, atau menyebut hasil dari pilihan itu, adalah spoiler (1). Menyebut bahwa suatu pilihan "penting" tanpa menyebut apa hasilnya adalah bukan spoiler (0), dan ini diatur oleh T1.
- **Jumlah ending:** menyebut bahwa ada banyak ending, atau angka berapa jumlahnya, bukan spoiler (0), sesuai S4. Labeling tidak boleh bergantung pada klaim jumlah ending, karena klaim itu belum diverifikasi.
- **G1 (mekanik tanpa dampak cerita):** tips atau mekanik yang hanya memengaruhi gameplay, seperti uang hangus, harga, kerusakan kendaraan, cuaca, atau cara menembak, bukan spoiler (0).
- **Batas:** jika komentar menyebut mekanik tanpa jelas apakah ada dampak ceritanya, pelabel memakai -1 dan menulis alasannya di `catatan`. Jangan menebak.
- Setiap label yang memakai S5 dicatat dengan kode S5 di `catatan`, agar bisa dianalisis terpisah.

**Ambigu (-1):** tidak bisa diputuskan dengan jelas. Tidak dipakai untuk latih, validasi, atau uji.

Aturan turunan:
- **S1 (nasib karakter):** menyebut apa yang terjadi pada karakter tertentu adalah spoiler.
- **S2 (twist):** mengungkap siapa atau apa yang tidak terduga adalah spoiler, meskipun tanpa nama.
- **S3 (hasil ending):** menyebut hasil ending tertentu adalah spoiler.
- **S4 (keberadaan dan nomor ending):** menyebut bahwa ada banyak ending, atau nomornya, bukan spoiler. Menghubungkan ending dengan nasib karakter adalah spoiler.
- **G1 (mekanik dan tips):** bukan spoiler, kecuali mengungkap peristiwa cerita.
- **T1 (twist tanpa isi):** menyebut adanya twist atau saran menyiapkan diri bukan spoiler.
- **R1 (reaksi dan harapan):** bukan spoiler.
- **Abu-abu:** jika pelabel tidak yakin, gunakan -1 dan tulis alasan di `catatan`.

Contoh lengkap ada di Lampiran A panduan. Contoh tersebut hanya untuk latihan pelabel.

---

## 11. Pengujian dan kualitas kode

- `pytest -q` lulus sebelum setiap commit.
- Setiap fungsi berlogika di `src/` punya minimal satu test.
- Gunakan type hint dan docstring singkat (bahasa Indonesia).
- Setiap skrip punya `if __name__ == "__main__":` dan `argparse`.
- Gunakan `logging`, bukan `print`, untuk pesan proses.
- Tetapkan seed untuk `random`, `numpy`, dan `torch`.

---

## 12. Git

```bash
git init
git add .gitignore CLAUDE.md README.md requirements.txt requirements.lock.txt src tests panduan kamus
git commit -m "Setup awal proyek"
```

Jangan `git add` `data/`, `models/`, atau `.env`. Periksa `git status` sebelum commit.

---

## 13. Cara Claude Code bekerja dengan proyek ini

1. Baca `CLAUDE.md` dan `README.md` sebelum mengubah apa pun.
2. Kerjakan satu tahap sekaligus. Di akhir tahap, jalankan test dan tampilkan ringkasan: file yang dibuat, jumlah data per split, dan hasil test.
3. Jika menemukan masalah di luar dokumen ini, berhenti dan tanyakan.
4. Jangan memperluas cakupan (metode ketiga, kanal lain, atau video di luar tabel) kecuali saya memintanya.
5. Setiap keputusan yang memengaruhi hasil (pilihan video, pembuangan data, ambang kappa, sampling, kamus) dicatat di README bagian "Keputusan".

---

## 14. Placeholder yang harus diisi saya

- `.env`: `YOUTUBE_API_KEY` dan `HASH_SALT`.
- `panduan/catatan_kejadian.md`: dibuat manual dari menonton setiap episode.
- `MODEL_BERT`: cek lisensi dan ketersediaan.
- Jumlah ending GTA 6: `TODO_VERIFIKASI` sampai ada sumber resmi.

---

## 15. Checklist

- [ ] `.env` ada dan tidak di-commit
- [ ] Pembagian video sesuai bagian 2 dan dibaca dari `.env`
- [ ] Data legacy dan `spoiler_example` tidak masuk pipeline (ada tes)
- [ ] `pelabel_awal.py` dan kolom `label_model` tidak dipakai
- [ ] Panduan label final dan catatan kejadian tersedia sebelum pelabelan
- [ ] Kappa ≥ 0,6 dilaporkan
- [ ] Split per video, assert tidak beririsan, tes split lulus
- [ ] Hyperparameter dipilih dengan data validasi saja
- [ ] Metrik kelas 1, rata-rata ± simpangan baku tiga seed, CI bootstrap, McNemar tersimpan
- [ ] Tidak ada seed atau model dipilih berdasarkan data uji
- [ ] Analisis kesalahan memisahkan galat label dan galat model
- [ ] Jumlah spoiler di `test.csv` dilaporkan sebagai keterbatasan
- [ ] README, pytest, dan requirements lock diperbarui
- [ ] Etika dan keterbatasan ditulis di laporan
