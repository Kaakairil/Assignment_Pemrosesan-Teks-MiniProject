# Panduan Label — Deteksi Spoiler Komentar GTA 6 (Windah Basudara)

**Versi: 2.1** — 2026-11-26. Menggantikan versi 1.0 dan 2.0. Setelah pelabelan
dimulai, panduan tidak boleh diubah tanpa mencatat ulang dan melabeli ulang
data terdampak (lihat README bagian Keputusan).

Sumber aturan: `docs/update/CLAUDE.md` bagian 10. Contoh di Lampiran A hanya
untuk latihan pelabel, bukan data latih.

---

## 1. Aturan umum untuk pelabel

1. Pelabel hanya boleh memakai **teks komentar + panduan ini + catatan kejadian**
   (`panduan/catatan_kejadian.md`). Pengetahuan pribadi tentang ending atau
   cerita game TIDAK boleh dipakai.
2. Label berdasarkan **isi komentar**, bukan sifat lain (panjang, jumlah like,
   siapa yang komentar). Timestamp di komentar TIDAK menentukan label —
   label mengikuti isi komentarnya.
3. Jika tidak yakin, gunakan **-1 (ambigu)** dan tulis alasannya di kolom
   `catatan`. Jangan menebak.
4. Label yang memakai aturan S5 dicatat dengan kode "S5" di `catatan`.

## 2. Definisi

**Spoiler (1):** mengungkap isi cerita atau konsekuensi cerita yang belum
diketahui pembaca yang belum menamatkan game: alur, akhir, nasib karakter,
twist, peristiwa penting, dan mekanik yang menentukan salah satunya (S5).
Berlaku juga untuk komentar yang menyebut momen spesifik di video.

**Bukan spoiler (0):** opini, pujian/kritik tanpa detail cerita, reaksi emosi,
harapan, pertanyaan, peringatan spoiler tanpa isi, dan tips/mekanik yang tidak
memengaruhi cerita (G1).

**Ambigu (-1):** tidak bisa diputuskan dengan jelas. Tidak dipakai untuk
latih, validasi, atau uji.

## 3. Aturan S (spoiler)

| Kode | Aturan | Contoh |
|---|---|---|
| S1 | Nasib karakter: menyebut apa yang terjadi pada karakter tertentu | "Pas karakter X mati aku nangis" |
| S2 | Twist: mengungkap siapa/apa yang tidak terduga, meski tanpa nama | "Ternyata yang sabun itu..." |
| S3 | Hasil ending: menyebut hasil ending tertentu | "Ending ke-7 yang dia dapet itu X" |
| S5 | Mekanik yang memengaruhi cerita: menyebut konsekuensi terhadap alur/ending/nasib karakter/hubungan/akses misi tersembunyi | "kalau trust meter di bawah 30%, Lucia ninggalin Jason" |
| S6 | Keputusan yang menentukan ending: pilihan dialog/misi/tindakan yang menentukan ending, atau hasilnya | "pilih X di misi terakhir, endingnya Y" |

## 4. Aturan G, T, R (bukan spoiler)

| Kode | Aturan | Contoh |
|---|---|---|
| G1 | Mekanik/tips tanpa dampak cerita: uang hangus, harga, kerusakan kendaraan, cuaca, cara menembak | "pake masker biar ga kena bintang" |
| S4 | Keberadaan/nomor ending: menyebut ada banyak ending atau nomornya | "ada 7 ending" / "ending ke-7 paling bagus" |
| T1 | Twist tanpa isi: menyebut adanya twist atau saran menyiapkan mental | "ada twist di akhir" (tanpa isi) / "siapin mental buat misi final" |
| R1 | Reaksi dan harapan | "keren banget" / "semoga lucia selamat" |

**Jumlah ending:** menyebut bahwa ada banyak ending atau angkanya = 0 (S4).
Labeling tidak boleh bergantung pada klaim jumlah ending (belum terverifikasi).

## 5. Kasus batas

- **Tips + konsekuensi cerita = SPOILER (S5).** Contoh yang WAJIB dilabeli 1:
  - "kalo uang 2 juta, endingnya bakal bagus / dapat true ending" → 1 (S5: mekanik uang + hasil ending)
  - "tembak kaki aja biar NPC/polisi ga mati, kalau ga begitu di ending Lucia dan Jason mati" → 1 (S5, sekaligus S1: menyebut nasib karakter di ending)
- Bandingkan: "pake masker biar ga kena bintang" → 0 (G1, tanpa dampak cerita).

- **Mekanik, tidak jelas dampak ceritanya** → -1 + alasan di `catatan`. Jangan menebak.
- **"Jangan spoiler ya, tapi endingnya..." lalu isinya** → 1 (isinya spoiler).
- **"Jangan spoiler!" saja** → 0.
- **Komentar timestamp + isi:** label mengikuti isi ("1:53:05 boros duit
  mana jelek motif nya" → 0, opini tanpa cerita).
- **Ending dibandingkan dengan trailer tanpa detail** → 0; dengan detail → 1.

## 6. Lampiran A — Contoh latihan

| Contoh | Label | Kode |
|---|---|---|
| "Endingnya keren banget" | 0 | R1 |
| "Endingnya bikin nangis" | 0 | R1 |
| "Pas karakter X mati aku nangis" | 1 | S1 |
| "Ending Windah beda sama yang di trailer" | 0 | R1 |
| "Ending ke-7 yang paling bagus" | 0 | S4 |
| "Ending ke-7 yang dia dapet itu X" | 1 | S3 |
| "Jangan spoiler ya, tapi endingnya..." lalu isinya | 1 | S3 |
| "Jangan spoiler!" saja | 0 | T1 |
| "Di menit 2:30 ternyata..." lalu menjelaskan | 1 | S2 |
| "pake masker biar ga kena bintang" | 0 | G1 |
| "kalo uang 2 juta, endingnya bakal bagus / dapat true ending" | 1 | S5 |
| "tembak kaki aja biar NPC ga mati, kalau ga, di ending Lucia dan Jason mati" | 1 | S5+S1 |
| "kalau trust meter di bawah 30%, Lucia ninggalin Jason" | 1 | S5 |
| "pilih X di misi terakhir, endingnya Y" | 1 | S6 |
| "siapin mental buat misi final" | 0 | T1 |

## 7. Lampiran B — Tes pemahaman pelabel

Sebelum pelabelan penuh, setiap pelabel melabeli **5 baris acak** dari
`data/spoiler_example/contoh_200_spoiler.csv` (tanpa melihat kolom `label`)
dengan panduan ini. Target: **minimal 4 dari 5 benar** menurut panduan 2.1.

Catatan: label di file contoh bertentangan dengan aturan G1 pada kategori
"Bocoran Mekanik" dan "Bocoran Misi" (tips/mekanik dilabeli `spoiler`).
Jika ketidakcocokan terjadi, **panduan 2.1 yang menentukan label yang benar** —
perbedaan itu adalah temuan yang dicatat, bukan kesalahan pelabel.
