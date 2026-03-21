# debug/generate_final_outputs.py

import pandas as pd
import os
import json

# -------------------------------
# File input dari batch
# -------------------------------
batch_file = "pipeline_results_dummy.xlsx"

if not os.path.exists(batch_file):
    raise FileNotFoundError(f"Batch file '{batch_file}' tidak ditemukan. Jalankan pipeline_batch_experiment.py dulu.")

# -------------------------------
# Load batch results
# -------------------------------
df = pd.read_excel(batch_file)

# -------------------------------
# Optional Postprocessing / Rule-based fix
# -------------------------------
# Contoh: Likelihood / Impact harus 1-10
for col in ["Likelihood (1-10)", "Impact (1-10)"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(5)  # default dummy
    df[col] = df[col].clip(1,10)  # pastikan 1-10

# Risk Priority fix: Low/Med/High
df["Risk Priority"] = df["Risk Priority"].apply(lambda x: x if str(x) in ["Low","Med","High"] else "Med")

# -------------------------------
# Simpan Excel final
# -------------------------------
excel_output = "final_outputs.xlsx"
df.to_excel(excel_output, index=False)
print(f"Final output saved to '{excel_output}'")

# -------------------------------
# Simpan JSON
# -------------------------------
json_output = "final_outputs.json"
df.to_json(json_output, orient="records", indent=2)
print(f"Final output also saved to '{json_output}'")