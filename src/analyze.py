"""
src/analyze.py
--------------
Modul Analisis Data, Statistik Deskriptif, dan Visualisasi.
Fokus pada EDA (Exploratory Data Analysis) hasil prediksi LLM, termasuk:
1. Statistik Deskriptif
2. Distribusi Risiko (Likelihood, Impact, Priority)
3. Matriks Risiko (Risk Matrix Heatmap 2D)
4. Kemiripan Teks / Deteksi Redundansi (TF-IDF Cosine Similarity)
"""

import os
import re
import warnings
import pandas as pd
import numpy as np

# Gunakan backend 'Agg' agar matplotlib bisa berjalan di server/Colab tanpa GUI (mencegah crash)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi tema plot yang estetik dan profesional
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore", category=FutureWarning)

def get_numeric_column(df, keyword):
    """Mencari nama kolom berdasarkan keyword (anti-hardcode)."""
    return next((col for col in df.columns if keyword.lower() in col.lower()), None)

def safe_to_numeric(series):
    """
    Ekstrak angka secara aman menggunakan Regex.
    Menyelamatkan program jika LLM berhalusinasi (misal menjawab '8 (Tinggi)' menjadi 8.0).
    """
    return pd.to_numeric(series.astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

def show_statistics(df):
    """Menampilkan ringkasan statistik deskriptif dari data."""
    if df is None or df.empty:
        print("⚠️ Data kosong, tidak dapat menampilkan statistik.")
        return

    print("\n" + "="*60)
    print("📊 STATISTIK DESKRIPTIF DATA RISIKO")
    print("="*60)
    print(f"Total Baris (Risiko) : {len(df)}")
    print(f"Total Kolom Atribut  : {len(df.columns)}")
    
    # Statistik Numerik
    l_col = get_numeric_column(df, 'likelihood')
    i_col = get_numeric_column(df, 'impact')
    
    num_cols = []
    if l_col: num_cols.append(l_col)
    if i_col: num_cols.append(i_col)

    if num_cols:
        df_num = df[num_cols].copy()
        for col in num_cols:
            df_num[col] = safe_to_numeric(df_num[col])
        print("\n📈 Distribusi Skoring Numerik (1-10):")
        print(df_num.describe().round(2).to_string())

    # Statistik Kategorikal
    cat_cols = ["Risk Priority (low, med, high)", "Project Stage", "Project Category", "Risk Owner"]
    print("\n📋 Top Kategori Dominan:")
    for col in cat_cols:
        if col in df.columns:
            top_vals = df[col].value_counts().head(3)
            print(f"  🔹 {col}:")
            for name, count in top_vals.items():
                print(f"      - {name}: {count}")
    print("="*60 + "\n")

def analyze_distribution(df, output_dir):
    """Membuat visualisasi distribusi bar chart dan histogram KDE."""
    if df is None or df.empty: return
    os.makedirs(output_dir, exist_ok=True)
    
    print("📈 Merender plot distribusi...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df.copy()

    # 1. Bar Chart Risk Priority
    p_col = get_numeric_column(df_plot, 'priority') or "Risk Priority (low, med, high)"
    if p_col in df_plot.columns:
        df_plot[p_col] = df_plot[p_col].astype(str).str.capitalize()
        order = ["Low", "Med", "High"]
        order = [x for x in order if x in df_plot[p_col].unique()]
        
        sns.countplot(data=df_plot, x=p_col, order=order, 
                      palette={"Low":"#2ecc71", "Med":"#f1c40f", "High":"#e74c3c"}, ax=axes[0])
        axes[0].set_title("Distribusi Prioritas Risiko", fontweight='bold')
        axes[0].set_xlabel("Tingkat Prioritas")
        axes[0].set_ylabel("Jumlah Risiko")
        
        # Tambah label angka di atas bar
        for p in axes[0].patches:
            axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    # 2. KDE Plot Likelihood vs Impact (ANTI-CRASH 0 VARIANCE)
    l_col = get_numeric_column(df_plot, 'likelihood')
    i_col = get_numeric_column(df_plot, 'impact')
    
    if l_col and i_col:
        df_plot[l_col] = safe_to_numeric(df_plot[l_col])
        df_plot[i_col] = safe_to_numeric(df_plot[i_col])
        
        l_data = df_plot[l_col].dropna()
        i_data = df_plot[i_col].dropna()
        
        if len(l_data) > 0 and len(i_data) > 0:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 🌟 FIX ERROR KEPADATAN 0 VARIANSI 🌟
                # Tambahkan noise super kecil jika angkanya kembar semua agar grafik tetap bisa digambar
                if l_data.nunique() <= 1: l_data = l_data + np.random.normal(0, 0.01, len(l_data))
                if i_data.nunique() <= 1: i_data = i_data + np.random.normal(0, 0.01, len(i_data))
                
                sns.kdeplot(l_data, color="skyblue", label="Likelihood", fill=True, ax=axes[1], warn_singular=False)
                sns.kdeplot(i_data, color="salmon", label="Impact", fill=True, ax=axes[1], warn_singular=False)
                
            axes[1].set_title("Density Plot: Likelihood vs Impact", fontweight='bold')
            axes[1].set_xlabel("Skor (1-10)")
            axes[1].legend()
            

def generate_risk_matrix(df, output_dir):
    """
    Membuat Matriks Risiko 2D (Standar ISO 31000).
    Grid Likelihood (Y) vs Impact (X). Sangat profesional untuk laporan kompetisi.
    """
    l_col = get_numeric_column(df, 'likelihood')
    i_col = get_numeric_column(df, 'impact')
    if not l_col or not i_col: return
    
    print("🔥 Merender Matriks Risiko (Risk Matrix Heatmap)...")
    os.makedirs(output_dir, exist_ok=True)
    
    df_num = pd.DataFrame()
    df_num['L'] = safe_to_numeric(df[l_col])
    df_num['I'] = safe_to_numeric(df[i_col])
    df_num = df_num.dropna()
    
    if df_num.empty: return
    
    # Buat grid 10x10 kosong
    matrix = pd.DataFrame(0, index=range(1, 11), columns=range(1, 11))
    
    # Hitung frekuensi risiko di tiap koordinat
    counts = df_num.groupby(['L', 'I']).size()
    for (l, i), count in counts.items():
        if 1 <= l <= 10 and 1 <= i <= 10:
            matrix.loc[int(l), int(i)] = count
            
    # Balik sumbu Y agar nilai 10 ada di atas
    matrix = matrix.sort_index(ascending=False)
    
    plt.figure(figsize=(7, 6))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt="g", cmap=cmap, linewidths=.5, linecolor='lightgray', cbar=True)
    
    plt.title('Risk Matrix Heatmap (Frekuensi Kejadian)', pad=15, fontweight='bold')
    plt.ylabel('Likelihood (1-10)', fontweight='bold')
    plt.xlabel('Impact (1-10)', fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "risk_matrix_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ Tersimpan: {save_path}")

def analyze_text_similarity(df, output_dir, text_col="Risk Description"):
    """
    Fitur WOW (Advanced NLP): 
    Mendeteksi risiko yang mirip secara semantik (redundansi data) menggunakan TF-IDF.
    """
    if df is None or text_col not in df.columns: return
    
    texts = df[text_col].dropna().astype(str).tolist()
    if len(texts) < 2: return
    
    print(f"🤖 Menjalankan Analisis Redundansi Semantik (NLP) pada '{text_col}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Ambil sample (maksimal 20 agar heatmap tidak keramaian)
    sample_texts = texts[:min(20, len(texts))]
    ids = df["Risk ID"].tolist()[:len(sample_texts)] if "Risk ID" in df.columns else [f"R{i+1}" for i in range(len(sample_texts))]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        plt.figure(figsize=(8, 7))
        # Masking segitiga atas agar rapi
        mask = np.triu(np.ones_like(cosine_sim, dtype=bool))
        
        sns.heatmap(cosine_sim, mask=mask, annot=False, cmap="mako", vmin=0, vmax=1,
                    xticklabels=ids, yticklabels=ids)
        plt.title("Semantic Similarity Heatmap\n(Deteksi Redundansi Risiko)", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, "semantic_similarity_heatmap.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   ✅ Tersimpan: {save_path}")
    except ValueError:
        pass # Abaikan jika teks hanya berisi angka/simbol aneh

def run_analysis(df, output_dir=None):
    """
    Orkestrator utama untuk menjalankan seluruh pipeline analitik pasca-prediksi LLM.
    """
    if df is None or df.empty:
        print("⚠️ Data kosong, analisis dibatalkan.")
        return
        
    if output_dir is None:
        # Tentukan folder output secara dinamis
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "outputs", "analysis_plots"))
        
    print("\n🔍 MEMULAI ANALISIS DATA (EDA)...")
    show_statistics(df)
    analyze_distribution(df, output_dir)
    generate_risk_matrix(df, output_dir)
    
    if "Risk Description" in df.columns:
        analyze_text_similarity(df, output_dir, "Risk Description")
        
    print("\n✨ Seluruh proses pembuatan Plot Visualisasi selesai!")

# ==============================================================================
# DEBUG / TESTING LOKAL
# ==============================================================================
if __name__ == "__main__":
    print("--- 🛠️ Menguji Modul Analisis Visualisasi ---")
    
    # Dummy data hasil LLM dengan berbagai jebakan (misal '8 (High)' di numerik)
    dummy_df = pd.DataFrame({
        "Risk ID": ["R1", "R2", "R3", "R4", "R5", "R6"],
        "Risk Description": [
            "Heavy rain delays construction process.",
            "Extreme weather causes project delay.", # Redundan dengan R1
            "Budget cuts force material downgrade.",
            "IT server failure causes data loss.",
            "Funding withdrawn by stakeholders.",
            "Hardware breakdown causes project delay."
        ],
        "Project Stage": ["Construction", "Construction", "Design", "Operational", "Planning", "Operational"],
        "Project Category": ["Environmental", "Environmental", "Financial", "Technical", "Stakeholder", "Technical"],
        "Likelihood (1-10)": ["8", "7", "5", "3 (Low)", "4", "2.0"],
        "Impact (1-10)": [7, 8, 9, 10, 8, 8],
        "Risk Priority (low, med, high)": ["High", "High", "Med", "Med", "Low", "Low"]
    })
    
    test_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "debug_cache", "test_plots"))
    run_analysis(dummy_df, output_dir=test_out)