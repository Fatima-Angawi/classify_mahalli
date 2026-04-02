from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
import os
import sys
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.preprocessing import combine_texts, validate_preview

# ======== إعداد المفتاح والعميل ========
os.environ["OPENAI_API_KEY"] = ""  # حطي مفتاحك الجديد هنا
client = OpenAI()

PROMPT_TEMPLATE = """
You are a system that classifies digital products.

Labels:
0 = Allowed
1 = Suspicious
2 = Illegal

Rules:
- Code / voucher → Allowed
- License key → Allowed
- Team invite / shared access → Illegal
- Account بيع حساب → Illegal
- Lifetime subscription → Illegal
- Unclear → Suspicious

Return ONLY JSON:
{{
 "label": int,
 "confidence": float,
 "reason": string,
 "detected_features": [string]
}}

Product:
{input}
"""

# ======== قراءة البيانات وتنظيفها ========
input_csv = "data/unlabeled_data.csv"
output_csv = "labeled_data.csv"

df = pd.read_csv(input_csv)
df = combine_texts(df)

if not validate_preview(df):
    print("Aborting due to preview rejection.")
    exit(1)

results = []

# ======== المعالجة ========
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["text"]
    prompt = PROMPT_TEMPLATE.format(input=text)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0
        )

        try:
            output = response.output[0].content[0].text.strip()
        except:
            raise ValueError("Empty or malformed response")
    
        # ✅ استخراج JSON حتى لو فيه كلام زيادة
        match = re.search(r"\{.*\}", output, re.DOTALL)

        if not match:
             raise ValueError(f"No JSON found. Raw output: {output}")

        json_text = match.group()

    # ✅ تحويل إلى dict
        parsed = json.loads(json_text)

        results.append({
            "text": text,
            "label": parsed.get("label"),
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"),
            "features": parsed.get("detected_features"),
            "source": "llm"
        })

    except Exception as e:
        # إذا نفاد الرصيد أو خطأ 429 → نوقف المعالجة
        if "insufficient_quota" in str(e) or "429" in str(e):
            print("Warning: OpenAI API quota exceeded. Stopping further processing.")
            break
        else:
            print("Error:", e)
            results.append({
                "text": text,
                "label": None,
                "source": "error"
            })

# ======== حفظ النتائج ========
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Saved {len(results)} results to {output_csv}")