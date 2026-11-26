# Pemilihan Video — Seri "GTA 6 Gameplay" Windah Basudara

Tanggal dokumen: 2026-11-26. Metadata (views, durasi) diambil dari YouTube Data API
(`videos.list`, satu panggilan untuk 11 video) pada 2026-11-26. Angka views dan
durasi **berubah seiring waktu** — catat tanggal setiap kali dirujuk ulang.

## Kriteria

- Seluruh episode seri "GTA 6 Gameplay" Windah Basudara yang sudah diarsipkan
  di kanalnya: episode #1 sampai #11.
- Pembagian per video menurut **urutan episode** (bukan hasil model, bukan acak):
  latih = #1–#8, validasi = #9–#10, uji = #11 (lihat `docs/update/CLAUDE.md` bagian 2).
- Video di luar tabel ini tidak boleh masuk dataset final.

## Tabel video

| Ep | Video ID | Judul | Tayang | Views (2026-11-26) | Durasi |
|---|---|---|---|---|---|
| #1 | `45r3OyaxvFE` | GTA VI siap tamat 100% | 2026-11-18 | 16.017.735 | 4:03:52 |
| #2 | `QSPPyA1736U` | Jason kangen dan pengen ketemu Lucia | 2026-11-19 | 5.895.875 | 1:49:58 |
| #3 | `B2Bon2X1jPQ` | Ketemu Bigfoot dan ke tempat sepatu | 2026-11-19 | 4.907.644 | 2:02:14 |
| #4 | `03qhcshfeMY` | Jason dan Lucia difitnah | 2026-11-20 | 5.711.366 | 3:52:24 |
| #5 | `lF_QYSm2T-Q` | Ketemu bayi alien dan UFO | 2026-11-21 | 4.980.897 | 4:57:16 |
| #6 | `u_Lb6sh9N0M` | Ganti mobil dan menemukan topi Arthur | 2026-11-22 | 3.011.139 | 1:49:20 |
| #7 | `MWQtXTTAgeM` | Bertemu Red Hulk | 2026-11-22 | 2.180.375 | 1:21:27 |
| #8 | `z6sBvvg_004` | Akhirnya pergi ke Vice City | 2026-11-23 | 4.068.488 | 4:41:25 |
| #9 | `zYTPbx83wCQ` | Misteri ritual satanis | 2026-11-24 | 1.455.699 | 1:06:30 |
| #10 | `wJVeWiqMJ-M` | Buka portal rahasia dan baju Tommy Vercetti | 2026-11-24 | 1.952.057 | 3:12:52 |
| #11 | `5m-Xi8lgvWY` | Akhirnya misi epic dan rampok bank | 2026-11-25 | 2.411.134 | 4:43:55 |

## Catatan

- Pembagian dipilih berdasarkan **urutan episode**, bukan hasil model. Ini
  mencegah kebocoran antar split karena komentar pada episode lanjutan sering
  membahas kejadian episode sebelumnya.
- #10 tayang live pada **24 November 2026**; #11 tayang live beberapa jam
  sebelum pengambilan komentar. Komentar kedua video ini belum final — jumlah
  dan isi komentar masih berkembang beberapa hari setelah tayang. Keterbatasan
  ini dicatat di laporan.
- Pengambilan komentar episode #1–#10 dilakukan 2026-11-25 (file mentah
  `data/raw/komentar_<VIDEO_ID>_2026-11-25_<halaman>.json`). Episode #11 belum
  diambil pada saat dokumen ini ditulis.
- Data adalah **komentar video arsip** (commentThreads), bukan live chat.
