import os, glob, pandas as pd

def list_input_files(input_folder='inputs'):
    files = glob.glob(os.path.join(input_folder,'*'))
    excel_files = [f for f in files if f.lower().endswith(('.xlsx','.xls','.xlsm','.xlsb'))]
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    return excel_files, pdf_files

def extract_excel_data(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"[ERROR] Cannot read {file_path}: {e}")
        return None

def extract_all_excels(input_folder='inputs'):
    excel_files,_ = list_input_files(input_folder)
    dfs = {}
    for f in excel_files:
        df = extract_excel_data(f)
        if df is not None:
            dfs[os.path.basename(f)] = df
            print(f"[INFO] Loaded {os.path.basename(f)} with {len(df)} rows")
    return dfs

def format_df_to_llm_text(df,row_index=None):
    texts=[]
    if row_index is not None:
        texts.append(str(df.iloc[row_index].to_dict()))
    else:
        for _,row in df.iterrows():
            texts.append(str(row.to_dict()))
    return texts