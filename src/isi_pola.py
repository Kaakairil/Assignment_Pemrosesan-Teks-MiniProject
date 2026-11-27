"""Isi kolom pola di analisis_kesalahan.csv dari terminal.

CLI: python -m src.isi_pola
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import HASIL

POLA = ["sindiran", "bahasa_campuran", "teks_pendek", "hanya_nama_karakter",
        "menyebut_ending_tanpa_detail", "tips_mekanik", "lainnya"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # emoji di teks komentar
    parser = argparse.ArgumentParser(description="Isi kolom pola per baris.")
    parser.add_argument("--file", type=Path, default=HASIL / "analisis_kesalahan.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.file, encoding="utf-8-sig")
    df = df.reset_index(drop=True)
    df["baris_excel"] = df.index + 2

    sisa = df["pola"].isna() | (df["pola"].astype(str).str.strip() == "")
    if not sisa.any():
        print("Semua baris sudah terisi. Selesai.")
        return

    print(f"{int((~sisa).sum())} baris belum terisi.\n")
    print("Kategori:", ", ".join(f"{i+1}={p}" for i, p in enumerate(POLA)))
    print("(Enter tanpa nomor = lewati baris ini)\n")

    for idx, r in df[sisa].iterrows():
        kel = " [GALAT-LABEL]" if str(r.get("jenis_galat", "")) == "galat_label" else ""
        t = r["teks_bert"] if pd.notna(r["teks_bert"]) and str(r["teks_bert"]).strip() else r["teks_klasik"]
        t = str(t).replace("\n", " ")
        if len(t) > 100:
            t = t[:100] + "…"
        salah = "FN" if r["fn_a"] else "FP"
        print(f"[{r['baris_excel']}] A:{salah}{kel} {t}")
        jawab = input("pola: ").strip().lower()
        if jawab:
            pilih = [p for p in POLA if p.startswith(jawab) or jawab == str(POLA.index(p) + 1)]
            if len(pilih) == 1:
                df.at[idx, "pola"] = pilih[0]
            else:
                print(f"  tidak jelas — pilihan: {', '.join(POLA)} (baris dilewati)")
        print()

    df = df.drop(columns=["baris_excel"])
    df.to_csv(args.file, index=False, encoding="utf-8-sig")
    terisi = df["pola"].notna() & (df["pola"].astype(str).str.strip() != "")
    print(f"Terisi: {int(terisi.sum())}/{len(df)} — tersimpan di {args.file}")


if __name__ == "__main__":
    main()
