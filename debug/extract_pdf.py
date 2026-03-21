# Placeholder: debug/extract_pdf.py
import os, glob, fitz
from typing import List

def extract_pdf(file_path:str)->List[str]:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return []
    texts=[]
    try:
        doc=fitz.open(file_path)
        for page in doc:
            t=page.get_text('text').replace('\n',' ').strip()
            if t: texts.append(t)
        print(f"[INFO] Extracted {len(texts)} pages from {os.path.basename(file_path)}")
        return texts
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF {file_path}: {e}")
        return []

def extract_all_pdfs(input_folder='inputs')->dict:
    pdf_files=glob.glob(os.path.join(input_folder,'*.pdf'))
    return {os.path.basename(f):extract_pdf(f) for f in pdf_files}