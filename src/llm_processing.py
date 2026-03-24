"""
src/llm_processing.py
---------------------
Core Engine: Hybrid Architecture (Rule-Based Extraction + Holistic JSON LLM).
- Menggunakan Regex untuk mengekstrak data pasti (Likelihood, Impact, RBS, Life).
- Memanfaatkan Few-Shot Learning agar AI menjiplak gaya bahasa Juri.
- Menyelesaikan masalah "Errno 28: No space left on device" (I/O Optimization).
"""

import os
import re
import json
import hashlib
import threading
import difflib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

# Pastikan import few_shot_builder tersedia
try:
    from few_shot_builder import get_few_shots_for_column
except ImportError:
    def get_few_shots_for_column(col): return "[]"

# Load API Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

try:
    from token_tracking import log_api_usage, count_tokens
except ImportError:
    def log_api_usage(p, c): pass
    def count_tokens(t): return len(str(t).split())

# ==============================================================================
# CACHING SYSTEM (MEMORY OPTIMIZED - FIX ERRNO 28)
# ==============================================================================
CACHE_DIR = os.path.join(BASE_DIR, "..", "data", "debug_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "llm_holistic_cache.json")
LLM_CACHE = {}
CACHE_LOCK = threading.Lock()
CACHE_MODIFIED = False 

def load_cache():
    global LLM_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                LLM_CACHE = json.load(f)
        except Exception: LLM_CACHE = {}
load_cache()

def save_cache_to_disk():
    """HANYA dipanggil sekali di akhir proses oleh run_pipeline.py agar Harddisk tidak meledak."""
    global CACHE_MODIFIED
    if not CACHE_MODIFIED: return
    with CACHE_LOCK:
        try:
            # Menggunakan separator compact agar ukuran file mengecil 50%
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(LLM_CACHE, f, ensure_ascii=False, separators=(',', ':'))
            CACHE_MODIFIED = False
        except OSError as e: 
            print(f"\n⚠️ Warning Disk Space Cache (Kosongkan Trash/Bin!): {e}")

def get_cache_key(prompt, text):
    return hashlib.md5(f"{model_name}_{prompt}_{text}".encode('utf-8')).hexdigest()

# ==============================================================================
# DYNAMIC CONSTRAINTS & HELPERS
# ==============================================================================
def load_valid_values():
    output_dir = Path(BASE_DIR).parent / "data" / "outputs"
    stages, categories, owners = set(), set(), set()
    if output_dir.exists():
        for file in output_dir.glob("*.xlsx"):
            try:
                df = pd.read_excel(file)
                if "Project Stage" in df.columns: stages.update(df["Project Stage"].dropna().astype(str).unique())
                if "Project Category" in df.columns: categories.update(df["Project Category"].dropna().astype(str).unique())
                elif "Risk Category" in df.columns: categories.update(df["Risk Category"].dropna().astype(str).unique())
                if "Risk Owner" in df.columns: owners.update(df["Risk Owner"].dropna().astype(str).unique())
            except Exception: continue
                
    if not stages: stages = {"Pre-construction", "Construction", "Operational", "Design", "Assembly and commissioning"}
    if not categories: categories = {"Planning", "Design", "Procurement", "Financial", "Technical", "Stakeholder", "Commercial"}
    if not owners: owners = {"Project Manager", "IT Manager", "Lead engineer", "Engineering mgmt", "Environmental"}
    
    return list({s.strip() for s in stages if s.strip()}), list({c.strip() for c in categories if c.strip()}), list({o.strip() for o in owners if o.strip()})

VALID_STAGES, VALID_CATEGORIES, VALID_OWNERS = load_valid_values()

def force_exact_match_fuzzy(llm_output, valid_list):
    out_lower = str(llm_output).strip().lower()
    if not out_lower or out_lower in ['none', 'null', 'unknown', 'na']: return "Unknown"
    for opt in valid_list:
        if opt.lower() == out_lower: return opt
    for opt in sorted(valid_list, key=len, reverse=True):
        if re.search(r'\b' + re.escape(opt.lower()) + r'\b', out_lower): return opt
    matches = difflib.get_close_matches(out_lower, [v.lower() for v in valid_list], n=1, cutoff=0.5)
    if matches: return next(v for v in valid_list if v.lower() == matches[0])
    return str(llm_output).strip() # Kembalikan tebakan asli AI jika Fuzzy gagal

# ==============================================================================
# 🌟 SUPER ALGORITHM: RULE-BASED EXTRACTION (DOMAIN KNOWLEDGE) 🌟
# ==============================================================================
def extract_explicit_values(target_text):
    """
    SAY NO TO LLM HALLUCINATION!
    Membaca data pasti dari raw teks. 
    Menerjemahkan kosakata khusus Dokumen 1 dan 2 ke format global.
    """
    explicit = {}
    t_lower = str(target_text).lower()
    
    # 1. Likelihood (Frequency / Likelihood / FRQ)
    freq_match = re.search(r'(frequency|likelihood|baseline frq|current_risk_likelihood)[\s]*[:=\-]?[\s]*(\d+)', t_lower)
    if freq_match: explicit['Likelihood'] = int(freq_match.group(2))
        
    # 2. Impact (Severity / Impact / SEV)
    sev_match = re.search(r'(severity|impact|baseline sev|current_risk_impact)[\s]*[:=\-]?[\s]*(\d+)', t_lower)
    if sev_match: explicit['Impact'] = int(sev_match.group(2))
        
    # 3. Project Stage (Life / Technology Life Phase)
    stage_match = re.search(r'(life|technology life phase|project stage)[\s]*[:=\-]?[\s]*([^|]+)', t_lower)
    if stage_match: 
        val = stage_match.group(2).strip()
        if len(val) > 2 and val != 'na': explicit['Project Stage'] = val.title()

    # 4. Project Category (RBS / Risk Category)
    cat_match = re.search(r'(rbs|rbs level 1|project category|risk category)[\s]*[:=\-]?[\s]*([^|]+)', t_lower)
    if cat_match: 
        val = cat_match.group(2).strip()
        if len(val) > 2 and val != 'na': explicit['Project Category'] = val.title()
        
    # 5. Risk Owner
    owner_match = re.search(r'(owner|risk owner)[\s]*[:=\-]?[\s]*([^|]+)', t_lower)
    if owner_match: 
        val = owner_match.group(2).strip()
        if len(val) > 2 and val != 'na': explicit['Risk Owner'] = val.title()
        
    return explicit

def calculate_priority_math(likelihood, impact):
    try:
        score = float(likelihood) * float(impact)
        if score <= 20: return "Low"
        elif score <= 50: return "Med"
        else: return "High"
    except Exception: return "Med"

# ==============================================================================
# HOLISTIC JSON CALL (DENGAN FEW-SHOT LEARNING)
# ==============================================================================
def process_single_risk(target_text, project_name=""):
    global CACHE_MODIFIED
    
    # 1. Ekstrak nilai pasti pakai Regex (Bypass AI)
    explicit_data = extract_explicit_values(target_text)
    
    # 2. Ambil contoh Few-Shot dari output Juri agar AI menjiplak gaya bahasanya
    try:
        few_shot_desc = get_few_shots_for_column("Risk Description")
        few_shot_mitigation = get_few_shots_for_column("Mitigating Action")
    except Exception:
        few_shot_desc = "[]"
        few_shot_mitigation = "[]"
    
    system_prompt = f"""You are an elite Data Scientist and Risk Expert.
Project Context: '{project_name}'

CRITICAL RULES:
1. "Project Stage" MUST be mapped to ONE of: {json.dumps(VALID_STAGES)}
2. "Project Category" MUST be mapped to ONE of: {json.dumps(VALID_CATEGORIES)}
3. "Risk Owner" MUST be mapped to ONE of: {json.dumps(VALID_OWNERS)}

LEARN FROM THESE GOLDEN EXAMPLES FOR SUMMARIZATION:
Risk Description Examples: {few_shot_desc[:400]}...
Mitigating Action Examples: {few_shot_mitigation[:400]}...

OUTPUT ONLY A VALID JSON OBJECT:
{{
    "Risk ID": "Extract or invent short ID (e.g. R1)",
    "Risk Description": "1 concise sentence summary matching the golden examples style",
    "Project Stage": "...",
    "Project Category": "...",
    "Risk Owner": "...",
    "Mitigating Action": "1 concise sentence summary matching the golden examples style",
    "Likelihood": <integer 1-10>,
    "Impact": <integer 1-10>
}}"""

    user_payload = f"--- TARGET RISK DATA ---\n{target_text}"

    cache_key = get_cache_key(system_prompt, user_payload)
    parsed_json = {}
    
    with CACHE_LOCK:
        if cache_key in LLM_CACHE:
            parsed_json = LLM_CACHE[cache_key]

    if not parsed_json and client:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.1, 
                max_tokens=300
            )
            raw_ans = response.choices[0].message.content.strip()
            raw_ans = re.sub(r"^```json\s*", "", raw_ans)
            raw_ans = re.sub(r"^```\s*", "", raw_ans)
            raw_ans = re.sub(r"\s*```$", "", raw_ans)
            
            parsed_json = json.loads(raw_ans.strip())
            
            if hasattr(response, 'usage') and response.usage and getattr(response.usage, 'prompt_tokens', 0) > 0:
                log_api_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            else:
                log_api_usage(count_tokens(system_prompt + user_payload), count_tokens(raw_ans))
                
            with CACHE_LOCK:
                LLM_CACHE[cache_key] = parsed_json
                CACHE_MODIFIED = True 
                
        except Exception as e:
            print(f"❌ LLM API Error: {e}")

    return _post_process_hybrid(parsed_json, explicit_data)

def _post_process_hybrid(parsed_json, explicit_data):
    """Menyatukan hasil LLM dengan Data Eksplisit (RegEx) + Hitung Matematika."""
    results = {}
    
    results["Risk ID"] = parsed_json.get("Risk ID", "R-UNK")
    results["Risk Description"] = parsed_json.get("Risk Description", "")
    results["Mitigating Action"] = parsed_json.get("Mitigating Action", "")
    
    # 🌟 PENGGABUNGAN DOMAIN KNOWLEDGE (Eksplisit Mengalahkan LLM) 🌟
    final_stage = explicit_data.get("Project Stage", parsed_json.get("Project Stage", ""))
    final_category = explicit_data.get("Project Category", parsed_json.get("Project Category", ""))
    final_owner = explicit_data.get("Risk Owner", parsed_json.get("Risk Owner", ""))
    
    results["Project Stage"] = force_exact_match_fuzzy(final_stage, VALID_STAGES)
    results["Project Category"] = force_exact_match_fuzzy(final_category, VALID_CATEGORIES)
    results["Risk Owner"] = force_exact_match_fuzzy(final_owner, VALID_OWNERS)
    
    # MURNI DOMAIN KNOWLEDGE MATEMATIKA
    final_l = explicit_data.get("Likelihood", parsed_json.get("Likelihood", 5))
    final_i = explicit_data.get("Impact", parsed_json.get("Impact", 5))
    
    try: final_l = int(float(final_l))
    except: final_l = 5
    try: final_i = int(float(final_i))
    except: final_i = 5
    
    final_l = max(1, min(10, final_l))
    final_i = max(1, min(10, final_i))
    
    results["Likelihood (1-10) (pre-mitigation)"] = final_l
    results["Impact (1-10) (pre-mitigation)"] = final_i
    results["Risk Priority (pre-mitigation)"] = calculate_priority_math(final_l, final_i)
    
    # PREDIKSI LOGIS POST-MITIGASI
    post_l = max(1, int(final_l * 0.6)) 
    post_i = max(1, int(final_i * 0.8)) 
    
    results["Likelihood (1-10) (post-mitigation)"] = post_l
    results["Impact (1-10) (post-mitigation)"] = post_i
    results["Risk Priority (post-mitigation)"] = calculate_priority_math(post_l, post_i)
    
    return results