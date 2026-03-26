"""
src/extract_data.py
-------------------
Modul ekstraksi data cerdas untuk file Excel dan PDF.
Dilengkapi dengan 'Smart Target Detection' untuk Excel dan 'AI-Driven Parsing' untuk PDF.
Sangat tangguh (robust) terhadap perubahan layout dokumen, baris kosong (ghost rows), dan merged cells.
"""

import os
import re
import json
import warnings
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Sembunyikan warning bawaan pandas/openpyxl agar terminal tetap bersih
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.worksheet._reader")
# Optimasi untuk pandas versi terbaru
pd.set_option('future.no_silent_downcasting', True)

# Load PyMuPDF untuk pemrosesan PDF
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("⚠️ PyMuPDF (fitz) belum di-install. Jalankan: pip install PyMuPDF")

# Setup LLM Client (DeepSeek) khusus untuk membantu Parsing layout tabel PDF
load_dotenv('../../.env')
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None


# ==============================================================================
# 1. SMART EXCEL EXTRACTOR (Anti-Hardcode)
# ==============================================================================
def extract_excel_data(filepath, header_row_count=1):
    """
    Mengekstrak data tabular dari file Excel yang kompleks.
    Menggunakan algoritma 'Structural Density + Keyword Scoring' untuk 
    secara otomatis mendeteksi baris header tabel.
    """
    try:
        xls = pd.ExcelFile(filepath)
    except Exception as e:
        print(f"❌ Error membuka Excel file {filepath}: {e}")
        return None

    best_sheet = None
    best_header_row_idx = -1
    highest_score = -1
    
    # Kata kunci spesifik yang biasanya menjadi kolom Risk Register
    keywords = ['risk', 'description', 'impact', 'likelihood', 'probability', 'owner', 'action', 'category', 'status', 'mitigation', 'severity', 'id', 'ref']
    
    # --- FASE 1: Deteksi Header (Scoring) ---
    for sheet_name in xls.sheet_names:
        df_tmp = pd.read_excel(xls, sheet_name=sheet_name, header=None).head(50)
        if df_tmp.empty:
            continue
            
        for idx, row in df_tmp.iterrows():
            non_null_cells = [val for val in row.values if pd.notna(val) and str(val).strip() != '']
            fill_count = len(non_null_cells)
            
            # Jika baris terlalu sepi (misal hanya judul laporan/cover), lewati
            if fill_count < 3: 
                continue
                
            keyword_matches = 0
            string_cells = [val for val in non_null_cells if isinstance(val, str)]
            for val in string_cells:
                # Abaikan paragraf panjang (itu ciri khas isi data, bukan nama header)
                if len(val) > 60: 
                    continue
                cell_lower = val.lower()
                for kw in keywords:
                    if kw in cell_lower:
                        keyword_matches += 1
                        
            # Formula Skor: Kepadatan + (Bobot Keyword x 50) + Bonus Tipe Data - Penalti Kedalaman Baris
            score = fill_count + (keyword_matches * 50)  
            if len(string_cells) == fill_count:
                score += 10
            score -= idx 
            
            if score > highest_score:
                highest_score = score
                best_sheet = sheet_name
                best_header_row_idx = idx

    if highest_score < 10:
        print(f"⚠️ Warning: Tidak ditemukan tabel berstruktur valid di {os.path.basename(filepath)}")
        return None

    # --- FASE 2: Ekstraksi dan Perakitan Header Bertingkat (Merged Cells) ---
    df_full = pd.read_excel(xls, sheet_name=best_sheet, header=None)
    df_header = df_full.iloc[best_header_row_idx : best_header_row_idx + header_row_count].copy()
    df_data = df_full.iloc[best_header_row_idx + header_row_count:].copy()
    
    # Forward-fill nilai kosong secara horizontal untuk memperbaiki "Merged Cells" Excel
    header_vals = df_header.values.tolist()
    for row_idx in range(len(header_vals)):
        last_val = None
        for col_idx in range(len(header_vals[row_idx])):
            val = header_vals[row_idx][col_idx]
            if pd.isna(val) or str(val).strip() == '':
                header_vals[row_idx][col_idx] = last_val
            else:
                last_val = val
                
    df_header = pd.DataFrame(header_vals)
    new_headers = []
    
    # Gabungkan header bertingkat ke dalam 1 nama string dengan underscore
    for col in df_header.columns:
        components = [str(val).strip() for val in df_header[col].values if pd.notna(val) and str(val).strip() != '']
        clean_components = []
        for k in components:
            if not clean_components or clean_components[-1] != k:
                clean_components.append(k)
        
        col_name = "_".join(clean_components)
        if not col_name:
            col_name = f"Column_{col}"
        new_headers.append(col_name)
        
    df_data.columns = new_headers
    
    # --- FASE 3: Pembersihan Data (Ghost Rows) ---
    df_data = df_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    valid_rows = []
    for idx, row in df_data.iterrows():
        text_cols = 0
        for val in row.values:
            if pd.notna(val):
                val_str = str(val).strip()
                # Hindari baris siluman sisa rumus Excel yang berisi "0.0", "NaN", atau spasi
                if len(val_str) > 1 and val_str not in ['0.0', 'NaN', 'None']:
                    text_cols += 1
        # Syarat minimal ada 2 sel yang bermakna agar dianggap baris tabel valid
        valid_rows.append(text_cols >= 2) 
        
    df_data = df_data[valid_rows].drop_duplicates().reset_index(drop=True)
    return df_data


# ==============================================================================
# 2. AI-DRIVEN PDF EXTRACTOR & REGEX SPLITTER
# ==============================================================================
def raw_pdf_to_json(page_text):
    """Meminta bantuan LLM untuk merakit teks vertikal PDF menjadi Array JSON terstruktur."""
    from openai import OpenAI
    import os, json
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "[https://api.deepseek.com](https://api.deepseek.com)")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    
    if not api_key: return []
    local_client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = """You are a highly precise data parsing assistant.
The following is vertically extracted text from a Corporate Risk Register PDF.
Groups of values reading down the page represent columns.
Output exactly and ONLY a valid JSON array of objects representing these risks.
Keys to extract: Reference, Risk and effects, Mitigation, Risk Owner, Actions being taken.
Do not wrap your response in markdown fences."""
    try:
        response = local_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"--- PAGE TEXT ---\n{page_text}"}
            ],
            temperature=0.0
        )
        res = response.choices[0].message.content.strip()
        if res.startswith("```json"): res = res[7:]
        if res.startswith("```"): res = res[3:]
        if res.endswith("```"): res = res[:-3]
        return json.loads(res.strip())
    except Exception: return []


def split_risk_effects(df):
    """
    Trik Rahasia: Membelah kolom 'Risk and effects' menggunakan Regex
    agar pipeline prediksi utama (master LLM) lebih gampang menarik kesimpulan.
    """
    # Cari nama kolom yang mirip dengan "Risk and effects" (case-insensitive)
    target_col = next((col for col in df.columns if "risk" in col.lower() and "effect" in col.lower()), None)
    
    if not target_col:
        return df

    risks = []
    effects = []

    for text in df[target_col].fillna(""):
        text = str(text)
        # Cari pola 'Risk: [Teks A] Effect: [Teks B]'
        risk_match = re.search(r"risk[:\-]\s*(.*?)\s*effects?[:\-]", text, re.IGNORECASE)
        effect_match = re.search(r"effects?[:\-]\s*(.*)", text, re.IGNORECASE)

        risk = risk_match.group(1).strip() if risk_match else text
        effect = effect_match.group(1).strip() if effect_match else ""

        risks.append(risk)
        effects.append(effect)

    # Sisipkan hasil belahan persis di sebelah kolom asli
    insert_loc = df.columns.get_loc(target_col)
    df.insert(insert_loc, "Explicit Risk", risks)
    df.insert(insert_loc + 1, "Explicit Effect", effects)

    return df

def extract_pdf_data(filepath):
    """Fungsi utama untuk mengekstrak tabel dari PDF menjadi pandas DataFrame."""
    if fitz is None:
        raise ImportError("PyMuPDF belum terinstall! (pip install PyMuPDF)")
        
    doc = fitz.open(filepath)
    all_risks = []
    
    print(f"📄 Parsing layout PDF {os.path.basename(filepath)} menggunakan AI...")
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text or len(text.strip()) < 100:
            continue
            
        print(f"  -> Memproses Halaman {page_num + 1}/{len(doc)}...")
        page_risks = raw_pdf_to_json(text)
        
        if page_risks:
            all_risks.extend(page_risks)
            
    if not all_risks:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_risks)
    
    # Terapkan trik Regex Split setelah berhasil dikonversi ke DataFrame
    df = split_risk_effects(df)
    
    return df


# ==============================================================================
# 3. LLM TEXT FORMATTER
# ==============================================================================
def format_df_to_llm_text(df):
    """
    Mengubah row di DataFrame menjadi baris teks naratif (Key: Value).
    Format horizontal yang dipisahkan pipa (|) ini sangat dianjurkan untuk mencegah AI kehilangan konteks.
    """
    if df is None or df.empty:
        return []
        
    llm_texts = []
    for idx, row in df.iterrows():
        row_texts = []
        for col_name, val in row.items():
            clean_val = str(val).replace('\n', ' ').replace('\r', ' ').strip()
            if pd.notna(val) and clean_val != '' and clean_val.lower() not in ['nan', 'none']:
                row_texts.append(f"{col_name}: {clean_val}")
        
        # Gabungkan menjadi string utuh per baris Excel/PDF
        llm_texts.append(" | ".join(row_texts))
        
    return llm_texts


# ==============================================================================
# DEBUG / TESTING LOKAL
# ==============================================================================
if __name__ == "__main__":
    # Test path (Sesuaikan dengan lokasi file mentah kamu)
    test_excel = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "inputs", "1. IVC DOE R2 (Input).xlsx"))
    test_pdf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "inputs", "5. Corporate_Risk_Register (Input).pdf"))
    
    if os.path.exists(test_excel):
        print(f"\n--- Menguji Ekstraksi EXCEL: {os.path.basename(test_excel)} ---")
        df_ex = extract_excel_data(test_excel)
        if df_ex is not None:
            print(f"✅ Excel Shape: {df_ex.shape}")
            print("✅ Format String Teks LLM (Baris 1):")
            print(format_df_to_llm_text(df_ex.head(1))[0][:250] + "...\n")
            
    if os.path.exists(test_pdf):
        print(f"\n--- Menguji Ekstraksi PDF: {os.path.basename(test_pdf)} ---")
        df_pdf = extract_pdf_data(test_pdf)
        if not df_pdf.empty:
            print(f"✅ PDF Shape: {df_pdf.shape}")
            print(f"✅ Kolom yang berhasil ditarik: {list(df_pdf.columns)}")