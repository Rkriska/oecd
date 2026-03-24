"""
==============================================================================
🏆 OECD-NEA RISK REGISTER - SUPER HYBRID PIPELINE
==============================================================================
Skrip Utama (Master Orchestrator) untuk Lomba Ekstraksi & Prediksi Risiko.
Menggabungkan: Smart Extraction -> NLP Preprocessing -> Parallel LLM -> 
Deterministic Math -> Risk Matrix Analytics -> Auto-Format Excel.

CARA MENJALANKAN (Di Terminal):
python run_pipeline.py
"""

import os
import sys
import time
from tqdm.auto import tqdm # Progress bar visual untuk terminal

# Menambahkan folder src ke system path agar modul kita bisa di-import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# --- IMPORT SELURUH MODUL SENJATA RAHASIA KITA ---
from extract_data import extract_excel_data, extract_pdf_data, format_df_to_llm_text
from preprocess import preprocess_dataframe
from llm_processing import process_single_risk
from analyze import run_analysis
from save_results import format_and_save_final_excel, save_debug_data

# Coba import token tracking jika ada, jika tidak, bypass dengan aman
try:
    from token_tracking import print_token_summary
except ImportError:
    def print_token_summary(*args, **kwargs): pass

# ==============================================================================
# KONFIGURASI FOLDER
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
DEBUG_DIR = os.path.join(BASE_DIR, "data", "debug_cache")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis_plots")

# Pembuat folder otomatis agar program kebal dari error "Directory Not Found"
for folder in [INPUT_DIR, OUTPUT_DIR, DEBUG_DIR, ANALYSIS_DIR]:
    os.makedirs(folder, exist_ok=True)

def get_project_context(filename):
    """Menyuntikkan konteks spesifik ke AI agar paham domain proyek dari file."""
    fname = filename.lower()
    if "moorgate" in fname: return "Moorgate Crossrail"
    elif "corporate" in fname or "pdf" in fname: return "Corporate Risk Register"
    elif "digital" in fname or "it" in fname: return "Digital Security IT"
    elif "york" in fname: return "City of York Council"
    else: return "IVC DOE Energy"

def process_file(filename):
    """Fungsi orkestrasi untuk memproses 1 file secara komplit dari A - Z."""
    print("\n" + "="*70)
    print(f"🚀 MEMULAI PROSES FILE: {filename}")
    print("="*70)
    
    filepath = os.path.join(INPUT_DIR, filename)
    project_name = get_project_context(filename)
    
    # ---------------------------------------------------------
    # TAHAP 1: DATA EXTRACTION (Smart Detection & AI PDF Parsing)
    # ---------------------------------------------------------
    print("📥 [Tahap 1/5] Mengekstrak data mentah...")
    if filename.lower().endswith('.pdf'):
        df_raw = extract_pdf_data(filepath)
    elif filename.lower().endswith(('.xlsx', '.xls')):
        df_raw = extract_excel_data(filepath)
    else:
        print(f"⚠️ Format file {filename} tidak didukung (Bukan .xlsx atau .pdf).")
        return

    if df_raw is None or df_raw.empty:
        print(f"❌ Ekstraksi gagal atau file kosong: {filename}")
        return
        
    save_debug_data(df_raw, f"1_raw_{filename}", DEBUG_DIR)
    
    # ---------------------------------------------------------
    # TAHAP 2: DATA PREPROCESSING (Semantic Cleaning)
    # ---------------------------------------------------------
    print("🧹 [Tahap 2/5] Membersihkan Noise & Ghost Rows...")
    df_clean = preprocess_dataframe(df_raw)
    save_debug_data(df_clean, f"2_clean_{filename}", DEBUG_DIR)
    
    # Format baris DataFrame menjadi Teks Naratif (Key: Value) yang LLM-Friendly
    llm_texts = format_df_to_llm_text(df_clean)
    print(f"   📋 Siap dikirim ke LLM: {len(llm_texts)} baris berharga.")

    # ---------------------------------------------------------
    # TAHAP 3: AI PREDICTION (Parallel LLM & Deterministic Math)
    # ---------------------------------------------------------
    print(f"🧠 [Tahap 3/5] Memulai Prediksi AI Paralel (Multithreading)...")
    predicted_results = []
    
    # Gunakan TQDM untuk progress bar yang estetik di terminal
    for i in tqdm(range(len(llm_texts)), desc="Memprediksi Risiko", unit="baris"):
        text_payload = llm_texts[i]
        
        # OTAK UTAMA BEKERJA (Memproses 8 pertanyaan LLM serentak per baris)
        result_dict = process_single_risk(target_text=text_payload, project_name=project_name)
        predicted_results.append(result_dict)
        
        # Jeda ringan agar API DeepSeek tidak memblokir koneksi (Rate Limit)
        time.sleep(0.05) 
        
    # ---------------------------------------------------------
    # TAHAP 4: FORMATTING & SAVING EXCEL
    # ---------------------------------------------------------
    print("\n💾 [Tahap 4/5] Memformat Kolom Excel (Golden Template)...")
    # Ubah nama file menjadi "(Final)"
    output_filename = filename.replace("(Input)", "(Final)").replace(".pdf", ".xlsx")
    if "(Final)" not in output_filename:
        output_filename = f"Final_{filename.replace('.pdf', '.xlsx')}"
        
    final_filepath = os.path.join(OUTPUT_DIR, output_filename)
    df_final = format_and_save_final_excel(predicted_results, final_filepath)
    
    # ---------------------------------------------------------
    # TAHAP 5: DATA ANALYSIS & VISUALIZATION (Plotting Heatmap)
    # ---------------------------------------------------------
    print("📊 [Tahap 5/5] Membuat Visualisasi Analitik (EDA)...")
    if df_final is not None:
        plot_folder = os.path.join(ANALYSIS_DIR, project_name)
        run_analysis(df_final, output_dir=plot_folder)


# ==============================================================================
# MAIN EXECUTION TRIGER
# ==============================================================================
if __name__ == "__main__":
    # Banner Terminal Profesional
    print("""
    ███████╗██╗   ██╗██████╗ ███████╗██████╗     ██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗ 
    ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗    ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗
    ███████╗██║   ██║██████╔╝█████╗  ██████╔╝    ███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║
    ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗    ██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║
    ███████║╚██████╔╝██║     ███████╗██║  ██║    ██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝
    ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ 
                    [ OECD-NEA AI RISK REGISTER PIPELINE ]
    """)
    
    # Ambil semua file Excel dan PDF dari folder inputs (abaikan file sistem tersembunyi `~`)
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.xlsx', '.pdf')) and not f.startswith('~')]
    
    if not input_files:
        print(f"⚠️ Folder 'data/inputs/' KOSONG!")
        print("Silakan masukkan file soal lomba (Excel/PDF) ke dalam folder tersebut lalu jalankan ulang skrip ini.")
        sys.exit()
        
    start_global = time.time()
        
    # Eksekusi seluruh file secara berurutan
    for file in sorted(input_files):
        process_file(file)
        
    # 🌟 TAMBAHKAN BARIS INI (Penyelamat Harddisk Penuh / Errno 28) 🌟
    try:
        import llm_processing
        llm_processing.save_cache_to_disk()
    except Exception: pass

    print("\n" + "★"*70)
    print(f"🎉 SEMUA SOAL LOMBA SELESAI DIPROSES...")

    # Cetak Total Invoice / Biaya Token API (Fitur Khusus untuk Presentasi)
    print_token_summary(model="deepseek-chat")