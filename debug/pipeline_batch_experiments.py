# debug/pipeline_batch_experiment.py

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import extract_excel
import pipeline

# -------------------------------
# Folder input Excel
# -------------------------------
input_folder = "inputs"
dfs = extract_excel.extract_all_excels(input_folder)

# -------------------------------
# Hasil batch
# -------------------------------
all_results = []

# -------------------------------
# Loop semua file dan semua row
# -------------------------------
for fname, df in dfs.items():
    for idx in range(len(df)):
        # Format row → string siap pipeline
        row_text = extract_excel.format_df_to_llm_text(df, row_index=idx)[0]

        # Panggil pipeline dummy
        try:
            res = pipeline.process_single_risk(row_text)
        except Exception as e:
            # Fallback dummy jika error
            res = {col: "DUMMY" for col in [
                "Risk ID","Risk Description","Project Stage","Project Category",
                "Risk Owner","Mitigating Action","Likelihood (1-10)","Impact (1-10)","Risk Priority"
            ]}
        
        # Simpan info tambahan
        res['Source File'] = fname
        res['Row Index'] = idx

        # Tambahkan ke hasil batch
        all_results.append(res)

# -------------------------------
# Simpan hasil batch ke Excel
# -------------------------------
output_file = "pipeline_results_dummy.xlsx"
pd.DataFrame(all_results).to_excel(output_file, index=False)
print(f"Batch processing done. Saved to '{output_file}'")