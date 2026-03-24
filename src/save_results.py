"""
src/save_results.py
-------------------
Modul untuk menyimpan hasil prediksi LLM ke format Excel akhir.
Mengatur urutan kolom agar presisi 100% dengan template panitia juri,
serta fitur auto-saving cache untuk melacak data jika terjadi error.
"""

import os
import pandas as pd
from datetime import datetime

# Urutan kolom mutlak sesuai format panitia (Golden Template)
TARGET_COLUMNS = [
    "Date Added", 
    "Risk ID", 
    "Risk Description", 
    "Project Stage", 
    "Project Category", 
    "Risk Owner", 
    "Likelihood (1-10) (pre-mitigation)", 
    "Impact (1-10) (pre-mitigation)", 
    "Risk Priority (pre-mitigation)", 
    "Mitigating Action",
    "Likelihood (1-10) (post-mitigation)", # Dikosongkan jika tidak diprediksi AI
    "Impact (1-10) (post-mitigation)",     
    "Risk Priority (post-mitigation)"      
]

def save_debug_data(df, filename, debug_dir):
    """
    Menyimpan DataFrame sementara (cache) selama proses preprocessing.
    Berguna untuk mereview data mentah sebelum masuk ke otak LLM.
    """
    os.makedirs(debug_dir, exist_ok=True)
    if df is not None and not df.empty:
        # Bersihkan nama file agar aman dari spasi
        safe_name = filename.replace(" ", "_").replace(".xlsx", "").replace(".pdf", "")
        save_path = os.path.join(debug_dir, f"{safe_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"   💾 [Debug Cache] Tersimpan: {os.path.basename(save_path)}")

def format_and_save_final_excel(results_list, output_filepath):
    """
    Mengubah list of dictionaries dari LLM menjadi DataFrame,
    memaksa urutan kolom sesuai template panitia, menyuntikkan 'Date Added',
    dan menyimpannya ke format Excel (.xlsx).
    """
    if not results_list:
        print("⚠️ Warning: Tidak ada hasil prediksi untuk disimpan.")
        return None
        
    print(f"💾 Memformat & Menyimpan {len(results_list)} baris data...")
    
    # 1. Konversi output LLM ke DataFrame Pandas
    df_out = pd.DataFrame(results_list)
    
    # 2. Tambahkan kolom 'Date Added' (Tanggal hari ini otomatis)
    # Format panitia: DD-MMM-YY (contoh: 21-Feb-17)
    today_str = datetime.today().strftime('%d-%b-%y')
    if "Date Added" not in df_out.columns:
        df_out.insert(0, "Date Added", today_str)
    else:
        df_out["Date Added"] = today_str
    
    # 3. Pastikan semua kolom target ada. Jika LLM tidak membuatnya, isi kosong ("")
    for col in TARGET_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = ""
            
    # 4. Re-index / Paksa urutan kolom agar posisinya mutlak tidak bergeser
    df_out = df_out[TARGET_COLUMNS]
    
    # 5. Simpan ke Excel tanpa menyertakan nomor baris (index=False)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_out.to_excel(output_filepath, index=False)
    
    print(f"   ✅ [FINAL OUTPUT] Excel Lomba tersimpan di: {os.path.basename(output_filepath)}")
    return df_out