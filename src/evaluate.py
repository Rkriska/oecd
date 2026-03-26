"""
src/evaluate.py
---------------
Modul Evaluasi Performa AI (Benchmarking).
Membandingkan hasil Excel AI (Generated) dengan Excel Referensi (Kunci Jawaban Juri).
Metrik: Text Cosine Similarity, Exact Match Rate, dan Numeric MAE.
"""

import math
import re
import os
from collections import Counter
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def _normalize_text(value) -> str:
    if pd.isna(value): return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def _token_cosine_similarity(text_a: str, text_b: str) -> float:
    a = _normalize_text(text_a)
    b = _normalize_text(text_b)
    if not a and not b: return 1.0
    if not a or not b: return 0.0

    ca = Counter(a.split())
    cb = Counter(b.split())
    vocab = set(ca) | set(cb)
    dot = sum(ca[t] * cb[t] for t in vocab)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

def _to_float(value):
    try:
        if pd.isna(value): return None
        text = str(value).strip()
        if not text: return None
        return float(text)
    except Exception: return None

def load_dataframe(path: Path) -> pd.DataFrame:
    """Otomatis memuat data baik CSV maupun Excel."""
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Format file tidak didukung: {path}")

def _align_rows(gen_df: pd.DataFrame, ref_df: pd.DataFrame, gen_key: str, ref_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gen_key in gen_df.columns and ref_key in ref_df.columns:
        gen = gen_df.copy()
        ref = ref_df.copy()
        # Menyamakan format kunci ID agar bisa di-merge dengan aman
        gen["__k"] = gen[gen_key].astype(str).str.strip().str.lower()
        ref["__k"] = ref[ref_key].astype(str).str.strip().str.lower()
        merged = gen.merge(ref, on="__k", how="inner", suffixes=("_gen", "_ref"))
        
        if len(merged) > 0:
            gen_cols = [c for c in merged.columns if c.endswith("_gen")]
            ref_cols = [c for c in merged.columns if c.endswith("_ref")]
            gen_aligned = merged[gen_cols].copy()
            ref_aligned = merged[ref_cols].copy()
            gen_aligned.columns = [c[:-4] for c in gen_aligned.columns]
            ref_aligned.columns = [c[:-4] for c in ref_aligned.columns]
            return gen_aligned.reset_index(drop=True), ref_aligned.reset_index(drop=True)

    # Fallback jika nama kunci ID tidak selaras
    n = min(len(gen_df), len(ref_df))
    return gen_df.iloc[:n].reset_index(drop=True), ref_df.iloc[:n].reset_index(drop=True)

def _evaluate_pair(doc_name: str, gen_path: Path, ref_path: Path, gen_key: str, ref_key: str) -> dict:
    try:
        gen_df = load_dataframe(gen_path)
        ref_df = load_dataframe(ref_path)
    except Exception as e:
        print(f"❌ Error membaca file {doc_name}: {e}")
        return None

    gen_df, ref_df = _align_rows(gen_df, ref_df, gen_key, ref_key)

    # Menyamakan huruf besar-kecil pada nama kolom (case-insensitive matching)
    gen_cols_lower = {c.strip().lower(): c for c in gen_df.columns}
    ref_cols_lower = {c.strip().lower(): c for c in ref_df.columns}
    common_cols_lower = set(gen_cols_lower.keys()).intersection(set(ref_cols_lower.keys()))
    
    # Abaikan kolom ID dan Date dalam perhitungan performa teks
    cols_to_evaluate = [c for c in common_cols_lower if 'date' not in c and 'id' not in c and 'number' not in c]

    if not cols_to_evaluate:
        return {"Document": doc_name, "Rows": 0, "Text Cosine (%)": 0.0, "Exact Match (%)": 0.0, "Numeric MAE": None, "Overall Score (%)": 0.0}

    text_scores, num_abs_errors = [], []
    exact_hits, exact_total, num_within, num_total = 0, 0, 0, 0

    for col_lower in cols_to_evaluate:
        gen_col = gen_cols_lower[col_lower]
        ref_col = ref_cols_lower[col_lower]
        
        gen_series = gen_df[gen_col]
        ref_series = ref_df[ref_col]

        gen_num = gen_series.apply(_to_float)
        ref_num = ref_series.apply(_to_float)
        numeric_mask = gen_num.notna() & ref_num.notna()

        # Deteksi Kolom Numerik (Likelihood, Impact)
        if numeric_mask.sum() > (len(gen_series) * 0.5):
            for gv, rv in zip(gen_num[numeric_mask], ref_num[numeric_mask]):
                err = abs(gv - rv)
                num_abs_errors.append(err)
                if err <= 0.1: num_within += 1
                num_total += 1
        else:
            # Deteksi Kolom Teks (Description, Category, Owner)
            for gv, rv in zip(gen_series, ref_series):
                gtxt = _normalize_text(gv)
                rtxt = _normalize_text(rv)
                text_scores.append(_token_cosine_similarity(gtxt, rtxt))
                exact_hits += int(gtxt == rtxt)
                exact_total += 1

    text_cosine_avg = sum(text_scores) / len(text_scores) if text_scores else 1.0
    exact_match_rate = exact_hits / exact_total if exact_total else 1.0
    numeric_mae = sum(num_abs_errors) / len(num_abs_errors) if num_abs_errors else None

    numeric_score = 1.0 if numeric_mae is None else max(0.0, 1.0 - (numeric_mae / 10.0))
    overall_score = (0.5 * text_cosine_avg) + (0.25 * exact_match_rate) + (0.25 * numeric_score)

    return {
        "Document": doc_name,
        "Rows": len(gen_df),
        "Text Cosine (%)": round(text_cosine_avg * 100, 2),
        "Exact Match (%)": round(exact_match_rate * 100, 2),
        "Numeric MAE": round(numeric_mae, 4) if numeric_mae is not None else None,
        "Overall Score (%)": round(overall_score * 100, 2),
    }

def run_evaluating():
    # Setup Paths Dinamis
    BASE_DIR = Path(os.path.abspath(__file__)).parent.parent
    GEN_DIR = BASE_DIR / "data" / "outputs"
    REF_DIR = BASE_DIR / "data" / "reference_outputs"

    os.makedirs(REF_DIR, exist_ok=True)

    mappings = [
        {
            "doc": "Dokumen 1 (IVC DOE R2)",
            "gen": GEN_DIR / "1. IVC DOE R2 (Final).xlsx",
            "ref_base": "1. IVC DOE R2", 
            "gen_key": "Risk ID",
            "ref_key": "Risk ID",
        },
        {
            "doc": "Dokumen 2 (City of York)",
            "gen": GEN_DIR / "2. City of York Council (Final).xlsx",
            "ref_base": "2. City of York Council",
            "gen_key": "Risk ID",
            "ref_key": "Risk ID",
        },
        {
            "doc": "Dokumen 3 (Digital IT)",
            "gen": GEN_DIR / "3. Digital Security IT Sample Register (Final).xlsx",
            "ref_base": "3. Digital Security IT Sample Register",
            "gen_key": "Risk ID",
            "ref_key": "Number", # Dokumen 3 asli dari Juri memakai kolom ID bernama 'Number'
        },
    ]

    results = []
    print("\n" + "="*60)
    print("🎯 MEMULAI EVALUASI MATEMATIS AI (AI VS KUNCI JURI)")
    print("="*60)

    for item in mappings:
        gen_path = item["gen"]
        
        # Mengecek apakah file referensi Juri berekstensi .xlsx atau .csv
        ref_xlsx = REF_DIR / f"{item['ref_base']}.xlsx"
        ref_csv = REF_DIR / f"{item['ref_base']}.csv"
        ref_path = ref_xlsx if ref_xlsx.exists() else ref_csv if ref_csv.exists() else None

        if not gen_path.exists():
            print(f"⚠️ Skip {item['doc']}: File AI Generated tidak ditemukan -> {gen_path.name}")
            continue
        if not ref_path:
            print(f"⚠️ Skip {item['doc']}: Kunci Jawaban (Reference) tidak ditemukan!")
            continue
            
        result = _evaluate_pair(item["doc"], gen_path, ref_path, item["gen_key"], item["ref_key"])
        if result:
            results.append(result)
            print(f"✅ {item['doc']} | Rows: {result['Rows']} | Cosine: {result['Text Cosine (%)']}% | Exact: {result['Exact Match (%)']}% | MAE: {result['Numeric MAE']} | Overall: {result['Overall Score (%)']}%")

    if not results:
        print("❌ Tidak ada data yang dievaluasi. Pastikan file referensi asli ditaruh di data/reference_outputs/")
        return None

    df_results = pd.DataFrame(results)
    eval_out_path = GEN_DIR / "AI_Performance_Evaluation_Report.csv"
    df_results.to_csv(eval_out_path, index=False)
    print(f"\n💾 Laporan Benchmarking tersimpan di: {eval_out_path.name}")
    
    return df_results

if __name__ == "__main__":
    run_evaluating()