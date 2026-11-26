"""Loop pelabelan manual di terminal. Semua keputusan label dari manusia (Tahap 3/5).

Mode cepat: Enter = 0 (bukan spoiler), 1 = spoiler, -1 = ambigu.
Urutan bisa dipilih: --urut kunci (kata kunci cerita dulu), --urut pendek
(teks pendek dulu), --urut asli (urutan file).

Pakai: python -m src.label_manual --input data/labeled/lembar_label_a.csv
"""
import argparse
import textwrap
from pathlib import Path

import pandas as pd

LEBAR = 80

# Kata kunci yang sering muncul di komentar spoiler cerita (prioritas dulu).
KATA_KUNCI = [
    "ending", "tamat", "tewas", "mati", "killed", "dies", "twist", "ternyata",
    "rampok", "heist", "bank", "final", "terakhir", "lucia", "jason", "ritual",
    "portal", "versetti", "vercetti", "bocor", "spoiler", "penjara", "polisi",
    "difitnah", "fitnah", "kematian", "meninggal", "wafat", "copet", "nih cerita",
]


def pilih_label(jawaban: str) -> int | str | None:
    """Ubah input jadi label int, atau string aksi ('s' lewati, 'u' urung, 'q' keluar).

    None jika input tidak dikenal. String kosong = 0 (mode cepat).
    """
    j = jawaban.strip()
    if j == "":
        return 0
    if j in {"1", "0", "-1"}:
        return int(j)
    if j in {"s", "u", "q"}:
        return j
    return None


def urutkan(df: pd.DataFrame, belum: list[int], urut: str) -> list[int]:
    """Kembalikan urutan indeks baris yang belum dilabeli sesuai mode."""
    if urut == "asli":
        return belum
    if urut == "pendek":
        panjang = df.loc[belum, "teks"].astype(str).str.len()
        return df.loc[belum].assign(_p=panjang).sort_values("_p").index.tolist()
    if urut == "kunci":
        teks = df.loc[belum, "teks"].astype(str).str.lower()

        def skor(t: str) -> int:
            return -sum(1 for k in KATA_KUNCI if k in t)

        return sorted(belum, key=lambda i: (skor(teks[i]), len(str(teks[i]))))
    raise ValueError(f"Mode urut tidak dikenal: {urut}")


def main() -> None:
    """CLI: python -m src.label_manual --input ... --pelabel a --urut kunci"""
    parser = argparse.ArgumentParser(description="Pelabelan manual satu komentar per layar.")
    parser.add_argument("--input", type=Path, default=Path("data/labeled/lembar_label_a.csv"))
    parser.add_argument("--pelabel", default="", help="Penanda pelabel (mis. a atau b), disimpan di catatan")
    parser.add_argument("--urut", default="asli", choices=["asli", "pendek", "kunci"],
                        help="asli=urutan file, pendek=teks pendek dulu, kunci=kata kunci cerita dulu")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if "label" not in df.columns:
        raise SystemExit(f"Kolom 'label' tidak ada di {args.input}.")

    belum = df.index[df["label"].isna() | (df["label"].astype(str).str.strip() == "")].tolist()
    belum = urutkan(df, belum, args.urut)
    print(f"{args.input}: {len(df)} baris, {len(belum)} belum dilabeli (urut: {args.urut}).")
    if not belum:
        print("Semua sudah dilabeli. Selesai.")
        return

    print("Mode cepat: ENTER = 0. Ketik 1 = spoiler, -1 = ambigu, s = lewati, u = urung, q = keluar.")
    print("Aturan lengkap: panduan/panduan_label.md")
    urung: list[tuple[int, object]] = []  # (index, nilai_label_sebelumnya)

    i = 0
    while i < len(belum):
        idx = belum[i]
        teks = str(df.at[idx, "teks"])
        print("\n" + "=" * LEBAR)
        print(f"[{i + 1}/{len(belum)}]  id: {df.at[idx, 'id']}")
        print(textwrap.fill(teks, width=LEBAR))
        jawab = input("Label [ENTER=0 | 1=spoiler -1=ambigu | s u q]: ")

        pilih = pilih_label(jawab)
        if pilih is None:
            print("Input tidak dikenal, coba lagi.")
            continue
        if pilih == "q":
            break
        if pilih == "s":
            i += 1
            continue
        if pilih == "u":
            if not urung:
                print("Tidak ada yang bisa diurungkan.")
                continue
            idx_lama, nilai_lama = urung.pop()
            df.at[idx_lama, "label"] = nilai_lama
            i = belum.index(idx_lama)
            print(f"Kembali ke [{i + 1}/{len(belum)}].")
            continue

        urung.append((idx, df.at[idx, "label"]))
        df.at[idx, "label"] = pilih
        if args.pelabel:
            df.at[idx, "catatan"] = f"pelabel_{args.pelabel}"
        df.to_csv(args.input, index=False, encoding="utf-8-sig")
        i += 1

    df.to_csv(args.input, index=False, encoding="utf-8-sig")
    sudah = int((df["label"].astype(str).str.strip() != "").fillna(False).sum())
    print(f"\nTersimpan: {sudah} baris berlabel di {args.input}")
    print("Distribusi label:")
    print(df["label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
