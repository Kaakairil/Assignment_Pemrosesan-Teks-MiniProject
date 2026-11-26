"""Loop pelabelan manual di terminal. Semua keputusan label dari manusia (Tahap 3/5).

Pakai: python -m src.label_manual --input data/labeled/lembar_label_a.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import textwrap

LEBAR = 80


def pilih_label(jawaban: str) -> int | str | None:
    """Ubah input jadi label int, atau string aksi ('s' lewati, 'u' urung, 'q' keluar).

    None jika input tidak dikenal.
    """
    j = jawaban.strip()
    if j in {"1", "0", "-1"}:
        return int(j)
    if j in {"s", "u", "q"}:
        return j
    return None


def main() -> None:
    """CLI: python -m src.label_manual --input ... --pelabel a"""
    parser = argparse.ArgumentParser(description="Pelabelan manual satu komentar per layar.")
    parser.add_argument("--input", type=Path, default=Path("data/labeled/lembar_label_a.csv"))
    parser.add_argument("--pelabel", default="", help="Penanda pelabel (mis. a atau b), disimpan di catatan")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if "label" not in df.columns:
        raise SystemExit(f"Kolom 'label' tidak ada di {args.input}.")

    # Baris yang belum dilabeli.
    belum = df.index[df["label"].isna() | (df["label"].astype(str).str.strip() == "")].tolist()
    print(f"{args.input}: {len(df)} baris, {len(belum)} belum dilabeli.")
    if not belum:
        print("Semua sudah dilabeli. Selesai.")
        return

    print("Aturan label ada di panduan/panduan_label.md. Ketik angka untuk melabeli.")
    urung: list[tuple[int, object]] = []  # (index, nilai_label_sebelumnya)

    i = 0
    while i < len(belum):
        idx = belum[i]
        teks = str(df.at[idx, "teks"])
        print("\n" + "=" * LEBAR)
        print(f"[{i + 1}/{len(belum)}]  id: {df.at[idx, 'id']}")
        print(textwrap.fill(teks, width=LEBAR))
        jawab = input("Label [1=spoiler 0=bukan -1=ambigu | s=lewati u=urung q=keluar]: ")

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
