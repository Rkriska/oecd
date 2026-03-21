from few_shot_builder import get_few_shots_for_column
from openai import OpenAI
import os
from dotenv import load_dotenv
import concurrent.futures

# load_dotenv()
# api_key=os.getenv("DEEPSEEK_API_KEY")
# client=OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def call_llm(system_prompt,user_content):
    response=client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_content}],
        temperature=0.0,max_tokens=2048
    )
    return response.choices[0].message.content.strip()

def process_single_risk(target_text, debug=True):
    results={}
    columns=["Risk ID","Risk Description","Project Stage","Project Category","Risk Owner",
          "Mitigating Action","Likelihood (1-10)","Impact (1-10)","Risk Priority"]
    def call_col(col):
        return call_llm(get_few_shots_for_column(col), target_text)
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures={executor.submit(call_col,col):col for col in columns}
        for f in concurrent.futures.as_completed(futures):
            col=futures[f]
            try: val=f.result()
            except: val=""
            results[col]=val
            if debug: print(f"[DEBUG] {col}: {str(val)[:100]}")
    return results