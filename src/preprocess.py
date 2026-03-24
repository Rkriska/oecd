"""
src/preprocess.py
-----------------
Modul pembersihan dan pemformatan data (Data Preprocessing).
Fokus utama: 'Lightweight & Semantic Cleaning'.
"""

import re
import pandas as pd
import numpy as np

def clean_text(text):
    """
    Membersihkan teks dari noise tanpa merusak makna semantik atau tanda baca penting.
    """
    # 1. Tangani nilai kosong bawaan Pandas
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # 2. Tangani "Fake Null" (teks 'nan', 'none', 'null' hasil artefak konversi Excel/PDF)
    if text.strip().lower() in ['nan', 'none', 'nat', 'null', 'na', '0.0', '#n/a']:
        return ""
        
    # 3. Ganti karakter bullet point aneh hasil ekstraksi PDF (\uf0b7, \u2022, dll)
    text = re.sub(r'[\uf0b7\u2022\uf0a7]', '-', text)
    
    # 4. Hapus artefak khas ekstraksi Excel/PDF
    text = text.replace("_x000D_", " ").replace('\xa0', ' ')
    
    # 5. Ganti baris baru (\n), carriage return (\r), dan tab (\t) dengan satu spasi.
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # 6. Normalisasi spasi ganda atau lebih menjadi spasi tunggal
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    
    # 7. Jika setelah dibersihkan teks hanya berisi simbol sendirian, anggap kosong
    if text in ['-', '.', '_', '/', '\\', ':', ';']:
        return ""
        
    return text

def preprocess_dataframe(df):
    """
    Orkestrator untuk membersihkan seluruh DataFrame hasil ekstraksi.
    """
    if df is None or df.empty:
        print("⚠️ Warning: DataFrame kosong, proses cleaning dilewati.")
        return pd.DataFrame()

    print(f"🧹 Memulai preprocessing DataFrame... (Shape awal: {df.shape})")

    # 1. Bersihkan nama kolom dari enter dan spasi berlebih
    cleaned_cols = [clean_text(str(col)) if pd.notna(col) else f"Unnamed_{i}" for i, col in enumerate(df.columns)]

    # =====================================================================
    # 🌟 FIX ERROR VALUERROR: ANTI-CRASH NAMA KOLOM DUPLIKAT
    # =====================================================================
    new_cols = []
    seen = {}
    for col in cleaned_cols:
        if not col: col = "Unnamed" # Jika nama kolom kosong
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}") # Menjadi "Kolom_1", "Kolom_2"
        else:
            seen[col] = 0
            new_cols.append(col)
            
    df.columns = new_cols
    # =====================================================================

    # 2. Hapus kolom dan baris yang 100% NaN (mengantisipasi kolom/baris hantu Excel)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # 3. Terapkan clean_text() HANYA pada kolom bertipe teks/object
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    # 4. Filter ulang baris kosong pasca-cleaning (Penghancur Ghost Rows)
    df = df.replace("", np.nan)
    min_valid_cols = min(2, len(df.columns))
    df = df.dropna(thresh=min_valid_cols, axis=0)
    
    # 5. Kembalikan NaN menjadi string kosong "" (Format prompt LLM lebih suka string kosong)
    df = df.fillna("")

    # 6. Hapus duplikat persis jika ada
    df = df.drop_duplicates().reset_index(drop=True)
    
    print(f"✨ Preprocessing selesai! (Shape akhir: {df.shape})")
    
    return df