"""
src/save_results.py
-------------------
Filter Pipeline: Mengambil dictionary (Prediksi + Reasoning),
membuang Reasoning-nya, dan mengunci Format Excel sesuai Template Juri.
"""

import pandas as pd
from datetime import datetime
import os

def format_and_save_final_excel(llm_results_list, output_path):
    if not llm_results_list:
        raise ValueError("Data kosong, tidak ada yang bisa disimpan.")
        
    df = pd.DataFrame(llm_results_list)
    
    # 🎯 TARGET KOLOM MUTLAK UNTUK JURI (TANPA REASONING)
    target_columns = [
        "Risk ID",
        "Risk Description",
        "Project Stage",
        "Project Category",
        "Risk Owner",
        "Mitigating Action",
        "Likelihood (1-10) (pre-mitigation)",
        "Impact (1-10) (pre-mitigation)",
        "Risk Priority (pre-mitigation)",
        "Likelihood (1-10) (post-mitigation)",
        "Impact (1-10) (post-mitigation)",
        "Risk Priority (post-mitigation)"
    ]
    
    df_final = pd.DataFrame()
    
    # Saring data prediksi dan OTOMATIS MEMBUANG data Reasoning
    for col in target_columns:
        if col in df.columns:
            df_final[col] = df[col]
        else:
            df_final[col] = "" 
            
    # Tambahkan cap waktu (Date Added)
    df_final["Date Added"] = datetime.now().strftime("%Y-%m-%d")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_excel(output_path, index=False)
    
    return df_final