"""
run_pipeline.py
---------------
Master Orchestrator - Dilengkapi Auto-Cleanup Memory Manager.
Mengeksekusi 5 dokumen berurutan tanpa membebani storage / Errno 28.
"""

import os
import sys
import time
import glob
from IPython.display import clear_output
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath("")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "inputs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "outputs")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "data", "debug_cache")

# ==============================================================================
# MEMORY MANAGER (PENCEGAH ERRNO 28 MAC)
# ==============================================================================
def free_up_disk_space():
    """Sapu bersih file .csv sementara agar harddisk lega setiap ganti dokumen."""
    try:
        csv_files = glob.glob(os.path.join(DEBUG_DIR, "*.csv"))
        for f in csv_files:
            os.remove(f)
    except Exception: pass

free_up_disk_space()

# Impor Modul Core
from extract_data import extract_excel_data, extract_pdf_data, format_df_to_llm_text
from preprocess import preprocess_dataframe
from llm_processing import process_single_risk, save_cache_to_disk
from save_results import format_and_save_final_excel
from analyze import run_analysis

def process_file(filename):
    file_path = os.path.join(INPUT_DIR, filename)
    file_ext = filename.split('.')[-1].lower()
    base_name = os.path.splitext(filename)[0]
    final_excel_name = f"{base_name.replace(' (Input)', '')} (Final).xlsx"
    final_excel_path = os.path.join(OUTPUT_DIR, final_excel_name)
    clean_project_name = base_name.replace(" (Input)", "").replace("_", " ")
    
    print("\n" + "="*70)
    print(f"🚀 MEMULAI PROSES FILE: {filename}")
    print("="*70)

    # 1. EKSTRAKSI
    print("📥 [Tahap 1/5] Mengekstrak data mentah...")
    if file_ext in ['xlsx', 'xls']: df_raw = extract_excel_data(file_path)
    elif file_ext == 'pdf': df_raw = extract_pdf_data(file_path)
    else: return

    if df_raw is None or df_raw.empty:
        print(f"❌ Ekstraksi gagal: {filename}")
        return

    # 2. PREPROCESSING
    print("🧹 [Tahap 2/5] Membersihkan Noise & Ghost Rows...")
    df_clean = preprocess_dataframe(df_raw)
    llm_texts = format_df_to_llm_text(df_clean)
    print(f"   📋 Siap dikirim ke LLM: {len(llm_texts)} baris berharga.")

    # 3. AI PREDICTION DENGAN EXPLAINABLE AI
    print("🧠 [Tahap 3/6] Memulai Prediksi AI Paralel (Multithreading)...")
    final_results = []
    
    for text in tqdm(llm_texts, desc="Memprediksi Risiko"):
        res = process_single_risk(target_text=text, project_name=clean_project_name)
        final_results.append(res)
        
    # 🌟 NEW TAHAP 4: SAVE AUDIT CSV (PREDIKSI + ALASAN) 🌟
    print("\n📝 [Tahap 4/6] Menyimpan Log Audit Prediksi & Reasoning (CSV)...")
    csv_audit_name = f"{clean_project_name} (Prediction & Reasoning).csv"
    csv_audit_path = os.path.join(OUTPUT_DIR, csv_audit_name)
    try:
        import pandas as pd
        df_audit = pd.DataFrame(final_results)
        # Simpan utuh beserta kolom reasoning-nya
        df_audit.to_csv(csv_audit_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ [AUDIT LOG] Tersimpan: {csv_audit_name} (Siap untuk inspeksi Juri)")
    except Exception as e:
        print(f"   ⚠️ Gagal menyimpan CSV Audit: {e}")

    # 5. FORMATTING & SAVE FINAL EXCEL (Filter Kolom Target)
    print("💾 [Tahap 5/6] Menyaring Data ke Format Template Lomba Juri (Excel)...")
    try:
        # Fungsi ini otomatis akan menyaring & membuang kolom Reasoning
        df_final = format_and_save_final_excel(final_results, final_excel_path)
        print(f"   ✅ [FINAL OUTPUT] Excel tersimpan di: {final_excel_name}")
    except OSError as e:
        print(f"❌ Gagal menyimpan Excel! Memori Harddisk Penuh.")
        return

    # 6. VISUALISASI MATRIKS & GRAFIK
    print("📊 [Tahap 6/6] Membuat Visualisasi Analitik (EDA)...")
    plot_dir = os.path.join(OUTPUT_DIR, "analysis_plots", clean_project_name)
    try: run_analysis(df_final, output_dir=plot_dir)
    except OSError: pass

if __name__ == "__main__":
    clear_output()
    print("[ BATCH PROCESSING - FULL AUTO-PILOT ]\n")
    
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.xlsx', '.pdf')) and not f.startswith('~')]
    if not input_files:
        print(f"⚠️ Folder {INPUT_DIR} Kosong.")
    else:
        start_time = time.time()
        
        for file in sorted(input_files):
            try:
                process_file(file)
                free_up_disk_space() # Kosongkan RAM/Disk setiap selesai 1 file (MENCEGAH ERRNO 28)
            except Exception as e:
                print(f"❌ Error pada {file}: {e}")
                
        try: save_cache_to_disk() # Simpan cache 1x saja di paling akhir
        except: pass
        
        print("\n" + "★"*70)
        print(f"🎉 SEMUA SOAL LOMBA SELESAI DIPROSES DALAM {time.time() - start_time:.2f} DETIK! 🎉")
        print("★"*70)
        
        try:
            from token_tracking import print_token_summary
            print_token_summary(model=os.environ.get("DEEPSEEK_MODEL", "KoboiLLM"))
        except: pass