# Perbaikan Proyek: Deteksi Spoiler Komentar Video GTA 6 (Windah Basudara)

Dokumen ini berisi semua perbaikan yang masih diperlukan sebelum laporan final. Ikuti urutan bagian, karena setiap bagian menjadi syarat bagian berikutnya. Aturan teknis ada di `CLAUDE.md`.

Hasil evaluasi sebelumnya (F1 Metode A 0,667; Metode B 0,665 ± 0,049) dianggap **tidak valid**, karena berasal dari data dengan label heuristik dan pembagian acak. Jangan dipakai sebagai kesimpulan.

---

## 0. Keputusan wajib sebelum pelabelan

Bagian ini harus selesai sebelum bagian lain. Semua keputusan dicatat di `README.md` bagian "Keputusan" beserta tanggal dan alasan.

- [ ] **Sumber data.** Pilih salah satu:
  - Opsi 1: komentar YouTube episode #1–#11 lewat YouTube Data API (rancangan awal).
  - Opsi 2: komentar Instagram yang dikumpulkan manual. Syarat: ada kolom `sumber`, tidak ada identitas akun, tidak ada scraping otomatis, dan evaluasi dilaporkan per sumber.
  - Opsi 3: keduanya, dengan kolom `sumber` dan evaluasi terpisah.
- [ ] **Definisi spoiler versi 2.1.** Panduan di `CLAUDE.md` bagian 10 memuat aturan S5 (mekanik yang memengaruhi cerita) dan S6 (keputusan yang menentukan ending). Tulis alasan perubahan di laporan. Jelaskan bahwa jumlah ending (klaim komunitas, belum diverifikasi) tidak menjadi dasar label.
- [ ] **Verifikasi jumlah ending jika ingin disebut di laporan.** Cari sumber resmi Rockstar atau sumber yang bisa diaudit. Jika belum ada, tulis `TODO_VERIFIKASI`.
- [ ] **Hitung ulang proporsi kelas.** Dengan S5, proporsi spoiler kemungkinan naik. Hitung ulang pilot di bagian 5 dengan panduan 2.0.
- [ ] **Reset label versi 1.0.** Semua label dari panduan 1.0 dianggap tidak valid. Tulis di README bahwa label lama tidak dipakai, termasuk hasil tes pemahaman dan contoh pelabelan yang sudah dibuat.
- [ ] **Tanggal dan versi.** Tandai `panduan/panduan_label.md` dengan versi 2.0 dan tanggal, lalu jangan mengubahnya setelah pelabelan dimulai tanpa mencatat ulang.

**Selesai jika:** keempat keputusan tercatat di README sebelum ada satu pun label manusia.

## 1. Status saat ini

| Komponen | Status |
|---|---|
| Pipeline pengambilan, preprocessing, Metode A, Metode B, evaluasi | Sudah berjalan, 14 commit |
| Bug indeks di `evaluasi.py` | Sudah diperbaiki |
| Bug kolom teks di `analisis_kesalahan.py` | Sudah diperbaiki |
| `requirements.lock.txt` | Sudah dibuat |
| Pembagian video | Sudah final (11 episode, lihat `CLAUDE.md` bagian 2) |
| Data `data/spoiler_example/` | Sudah ada, hanya untuk latihan pelabel |
| Data lama 2.795 komentar | Masih ada di pipeline, **harus dipindahkan** ke `data/legacy/` |
| `src/pelabel_awal.py` dan kolom `label_model` | Masih ada, **harus dihapus** dari pipeline |
| Label manusia | **Belum dikerjakan** |
| Cohen's kappa | **Belum dihitung** |
| Hasil evaluasi final | **Belum ada** |

---

## 2. Bersihkan pipeline dari data dan label yang tidak valid

Ini langkah pertama karena tanpa ini semua hasil berikutnya tercemar.

- [ ] Pindahkan data lama 2.795 komentar ke `data/legacy/`.
- [ ] Tambahkan `data/legacy/` dan `data/spoiler_example/` ke `.gitignore`.
- [ ] Hapus `src/pelabel_awal.py` dari pipeline (`git rm`), dan pastikan tidak ada import ke modul tersebut.
- [ ] Hapus kolom `label_model` dari semua file yang dibaca `split.py` dan `evaluasi.py`.
- [ ] Pastikan `split.py` menolak baris yang tidak punya `video_id` dari tabel pembagian. Ini juga mencegah `data/spoiler_example/` masuk, karena ID-nya bilangan 1 sampai 40 tanpa `video_id`.
- [ ] Tulis `tests/test_split.py` dengan tes: baris legacy ditolak, baris `spoiler_example` ditolak, video di luar tabel ditolak, dan tidak ada irisan antar split.
- [ ] Tambahkan catatan di README bagian "Keputusan": data lama tidak dipakai dan alasannya.
- [ ] Pastikan `contoh_200_spoiler.csv` tidak masuk `train.csv`, `val.csv`, `test.csv`, atau `komentar_berlabel.csv`. Tambahkan tes ini di `tests/test_split.py`.
- [ ] Catat di README bahwa `contoh_200_spoiler.csv` memuat label yang bertentangan dengan aturan G1 (tips dan mekanik dilabeli `spoiler`), serta bahwa label-nya tidak berasal dari pelabel manusia.
- [ ] Gunakan `contoh_200_spoiler.csv` hanya untuk tes kode dan latihan pelabel (lihat bagian 6).

**Selesai jika:** `pytest` lulus dan `grep -r label_model src/` tidak menemukan apa pun.

---

## 3. Pemilihan video

- [ ] Buat `hasil/pemilihan_video.md` berisi:
  - Kriteria: seluruh episode "GTA 6 Gameplay" Windah Basudara yang sudah diarsipkan, #1 sampai #11, dibagi per video menurut urutan episode.
  - URL, ID, judul, dan tanggal tayang setiap episode.
  - Tanggal pengambilan, jumlah views, dan durasi, dengan catatan bahwa angka ini berubah seiring waktu.
- [ ] Catat bahwa pembagian dipilih berdasarkan urutan episode, bukan hasil model.
- [ ] Catat bahwa #10 tayang live pada 24 November 2026 dan #11 tayang live beberapa jam sebelum pengambilan. Komentar belum final sampai beberapa hari setelah tayang.

**Selesai jika:** file ada dan kriteria tertulis sebelum pengambilan final.

---

## 4. Konfigurasi dan pengambilan komentar

- [ ] Di `.env`, isi `TRAIN_VIDEO_IDS`, `VAL_VIDEO_IDS`, dan `TEST_VIDEO_IDS` sesuai `CLAUDE.md` bagian 2.
- [ ] `src/config.py` membaca ketiga daftar dari `.env`, lalu assert tidak beririsan.
- [ ] `src/ambil_komentar.py` menerima `--video-ids`, menyimpan file mentah per video, dan menambahkan `video_id` di setiap baris.
- [ ] Tambahkan awalan `yt_` pada kolom `id`.
- [ ] Pastikan kolom komentar aktif. Jika muncul `commentsDisabled`, catat di laporan.
- [ ] Catat tanggal dan jam setiap pengambilan di nama file mentah.
- [ ] Jelaskan di laporan bahwa data adalah komentar video arsip, bukan live chat.

**Selesai jika:** file mentah untuk kesebelas video ada, dengan tanggal dan `video_id`.

---

## 5. Pilot

Tujuan pilot adalah mengetahui apakah jumlah spoiler cukup sebelum pelabelan penuh.

- [ ] Ambil sekitar 300 komentar per video dari kesebelas episode.
- [ ] Saya labeli kasar 100 komentar untuk mengukur proporsi spoiler. Tahap ini hanya untuk perencanaan, bukan data penelitian.
- [ ] Jika proporsi spoiler di bawah sekitar 5 persen, pilih salah satu:
  - Ambil lebih banyak komentar.
  - Gunakan sampling terarah berbasis kata kunci misi, lokasi, karakter, atau "ternyata", "mati", "ending". Catat dan pisahkan dari sampel acak.
- [ ] Pastikan latih, validasi, dan uji masing-masing punya spoiler yang cukup. Jika `test.csv` punya kurang dari 20 spoiler, tulis sebagai keterbatasan utama.

**Selesai jika:** keputusan jumlah target dan strategi sampling tertulis dengan alasan.

---

## 6. Panduan label dan catatan kejadian

- [ ] Tulis `panduan/panduan_label.md` berisi definisi spoiler, bukan spoiler, dan ambigu, aturan S1 sampai S4, G1, T1, R1, aturan -1, dan contoh kasus. Sumber: `CLAUDE.md` bagian 10 dan Lampiran A yang sudah ada.
- [ ] Tambahkan aturan yang masih kosong:
  - Mekanik yang memengaruhi cerita (S5) adalah spoiler. Mekanik tanpa dampak cerita (G1) bukan spoiler. Jika tidak jelas, gunakan -1.
  - Komentar yang hanya menyebut "ada twist" tanpa isi (T1) bukan spoiler.
  - Saran menyiapkan mental sebelum misi final bukan spoiler.
  - Komentar yang menyebut menit atau timestamp: label mengikuti isi komentar, bukan timestamp-nya.
- [ ] Buat `panduan/catatan_kejadian.md` secara manual untuk episode #1 sampai #11: nama misi dan peristiwa penting. Ini dibuat dari menonton video oleh kamu, bukan dari Claude atau halaman YouTube.
- [ ] Jalankan Lampiran B (tes pemahaman) memakai `data/spoiler_example/` dengan dua pelabel. Target minimal 4 dari 5 benar sebelum pelabelan dimulai. Jika tidak tercapai, perbaiki panduan.
- [ ] Pakai `contoh_200_spoiler.csv` untuk latihan: minta pelabel memberi label tanpa melihat kolom `label`, lalu bandingkan. Hitung seberapa sering pelabel berbeda dengan kolom `label` pada kategori "Bocoran Mekanik" dan "Bocoran Misi". Perbedaan ini menunjukkan konflik dengan aturan G1.
- [ ] Putuskan dan tulis di panduan: apakah aturan G1 dipertahankan (mekanik bukan spoiler, sesuai panduan sekarang). Keputusan ini final, dan label contoh tidak boleh mengubah panduan.
- [ ] Jangan memasukkan `data/spoiler_example/` ke dataset apa pun.

**Selesai jika:** panduan final, catatan kejadian, dan hasil tes pemahaman tersedia sebelum pelabelan.

---

## 7. Pelabelan dan kesepakatan

- [ ] Siapkan `lembar_label_a.csv` dan `lembar_label_b.csv` dari `komentar_bersih.csv` dengan kolom `label` kosong.
- [ ] Dua pelabel menilai subset 100 komentar secara terpisah, dengan `SEED` tetap.
- [ ] Jalankan `src/kesepakatan.py`: hitung Cohen's kappa, buang -1, dan simpan tabel silang ke `hasil/kesepakatan.txt`.
- [ ] Jika kappa < 0,6, perbaiki panduan dan ulangi dengan subset baru. Jangan lanjut ke pelabelan penuh.
- [ ] Setelah kappa memenuhi ambang, labeli sisa data. Pertemukan dua pelabel untuk kasus yang berbeda dan tulis keputusan di `catatan`.

**Selesai jika:** kappa ≥ 0,6 dilaporkan dan label final tersedia untuk semua data latih, validasi, dan uji.

---

## 8. Split dan validasi

- [ ] Jalankan `src/split.py`. Buang label -1, lalu bagi per `video_id`:
  - Latih: #1 sampai #8
  - Validasi: #9 dan #10
  - Uji: #11
- [ ] Cetak jumlah dan proporsi kelas per split dan per video.
- [ ] Jalankan tes split. Semua harus lulus sebelum melanjutkan.

**Selesai jika:** tiga file split ada, tes lulus, dan jumlah spoiler per split tercatat.

---

## 9. Evaluasi

### 9.1 Metode A
- [ ] Pilih hyperparameter dengan melatih pada `train.csv` dan mengevaluasi pada `val.csv`. Jangan memakai `StratifiedKFold` acak.
- [ ] Kunci hyperparameter, latih ulang pada `train.csv`, lalu evaluasi **sekali** pada `test.csv`.

### 9.2 Metode B
- [ ] Kunci hyperparameter dengan early stopping pada `val.csv`.
- [ ] Jalankan tiga seed (`SEEDS_BERT`). Simpan prediksi per seed.
- [ ] Evaluasi sekali pada `test.csv` per seed.
- [ ] Jika GPU tidak tersedia, hentikan dan beri tahu saya.

### 9.3 Laporan metrik
- [ ] Precision, recall, dan F1 untuk kelas spoiler. Macro F1 sebagai pendukung.
- [ ] Metode B dilaporkan sebagai rata-rata ± simpangan baku tiga seed.
- [ ] **Hapus seluruh pemilihan "konsensus seed".** Jangan memilih seed berdasarkan data uji.
- [ ] CI 95 persen untuk F1 kelas spoiler dengan bootstrap pada `test.csv`.
- [ ] McNemar untuk A dan B pada prediksi uji yang sama.

### 9.4 Interpretasi
- [ ] Jika selisih F1 lebih kecil dari simpangan baku atau tumpang tindih dengan CI, tulis bahwa perbedaan tidak terdeteksi.
- [ ] Jelaskan bahwa hasil hanya dari satu episode uji dan bergantung pada jumlah spoiler yang kecil.

**Selesai jika:** `metrik.csv`, `bootstrap_ci.csv`, `mcnemar.txt`, dan confusion matrix tersedia.

---

## 10. Analisis kesalahan

- [ ] Gabungkan prediksi A dan B berdasarkan `id`.
- [ ] Pisahkan false negative dan false positive untuk masing-masing metode.
- [ ] Isi `pola` secara manual dengan kategori: sindiran, bahasa_campuran, teks_pendek, hanya_nama_karakter, menyebut_ending_tanpa_detail, tips_mekanik, lainnya.
- [ ] Isi `jenis_galat`: **galat label** (label bertentangan dengan panduan) atau **galat model**. Galat label tidak dihitung sebagai kegagalan model.
- [ ] Sertakan 5 sampai 10 contoh per pola tanpa identitas.

**Selesai jika:** analisis memisahkan galat label dan galat model, dengan contoh.

---

## 11. Dokumentasi, kode, dan laporan

- [ ] README memuat cara menjalankan pipeline, bagian "Keputusan", dan keterbatasan.
- [ ] `pytest -q` lulus.
- [ ] `requirements.lock.txt` sesuai lingkungan yang dipakai.
- [ ] `.env`, `data/`, dan `models/` tidak masuk git. Periksa `git status`.
- [ ] `laporan.md` berisi: pendahuluan, tinjauan pustaka (sumber akademik yang nyata), metodologi, hasil, analisis kesalahan, keterbatasan, kesimpulan, dan lampiran panduan.
- [ ] Klaim yang belum terverifikasi ditandai `TODO_VERIFIKASI`, misalnya jumlah ending.
- [ ] Tanggal pengambilan dan jumlah views dicantumkan dengan keterangan waktu.

---

## 12. Etika

- [ ] Tidak ada nama pengguna, ID penulis asli, atau tautan profil di data yang dibagikan.
- [ ] Komentar hanya untuk penelitian, tidak ditampilkan ulang dengan identitas.
- [ ] Sifat subjektif spoiler dijelaskan di laporan.
- [ ] Keterbatasan: satu kanal, satu seri, satu episode uji, dan label subjektif.

---

## 13. Urutan kerja

1. Bagian 2: bersihkan pipeline (legacy, `spoiler_example`, `label_model`).
2. Bagian 3 dan 4: pemilihan video dan pengambilan komentar.
3. Bagian 5: pilot.
4. Bagian 6: panduan dan catatan kejadian, lalu tes pemahaman.
5. Bagian 7: pelabelan dan kappa.
6. Bagian 8 dan 9: split dan evaluasi, sekali pada uji.
7. Bagian 10 sampai 12: analisis kesalahan, dokumentasi, dan laporan.

---

## 14. Checklist akhir

- [ ] Data legacy dan `spoiler_example` tidak masuk pipeline
- [ ] `pelabel_awal.py` dan `label_model` dihapus
- [ ] Pemilihan video tertulis sebelum pengambilan final
- [ ] Pilot dan jumlah target tercatat
- [ ] Panduan final dan catatan kejadian tersedia
- [ ] Kappa ≥ 0,6 dilaporkan
- [ ] Split per video, tes lulus
- [ ] Hyperparameter dipilih pada validasi
- [ ] Uji dievaluasi sekali
- [ ] Rata-rata ± simpangan baku tiga seed, CI bootstrap, McNemar
- [ ] Tidak ada pemilihan seed berdasarkan uji
- [ ] Galat label dipisahkan dari galat model
- [ ] Keterbatasan dan etika ditulis
