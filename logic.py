import os
import fitz  # PyMuPDF
import json
import re
import openai
from werkzeug.utils import secure_filename
#import pdfplumber
#import pytesseract
#from PIL import Image
#import io
#import pandas as pd
# pip install pdfplumber pytesseract pandas pillow tabulate

client = openai.OpenAI()


def evaluate_uploaded_file(file, upload_folder):
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    print("✅ File successfully uploaded.")

    classification_onc, explanation = run_oncology_classification_prompt(text)
    metadata = run_metadata_classification_prompt(text)
    overarching_rules_table_html = render_overarching_rules_table(text)

    classification_study_design = None
    study_design_explanation = None
    total_score = None
    study_score = None
    score_letter = None
    response_analysis = None
    flag = None
    rct_type = None
    rct_type_explanation = None
    study_score_explanation = None
    bonus_malus_delta = None
    bonus_malus_delta_real = 0
    total_score_cap_single_arm = None
    match = None
    bonus_malus_table_new = None
    rules_data = None

    if classification_onc == 0:
        print("🧬 Study is oncological")

    elif classification_onc == 1:
        print("🧪 Study is non-oncological")
        classification_study_design, study_design_explanation = run_study_design_classification_prompt(
            text)
        print("🧪 Study design classification:", classification_study_design)

        if classification_study_design == 0:  # RCT
            rct_type, rct_type_explanation = get_rct_type(text)
            study_score, study_score_explanation = get_rct_category(
                text, rct_type)
            if study_score is not None and study_score >= 1:
                rules_data = get_bonus_and_malus_rules(text)
                bonus_malus_table_new = render_bonus_malus_rules_table(
                    text, rules_data)
                bonus_malus_delta = calculate_bonus_malus_delta(rules_data)
                print(bonus_malus_delta)
                if bonus_malus_delta >= 1:
                    bonus_malus_delta_real = 1
                elif bonus_malus_delta <= -1:
                    bonus_malus_delta_real = -1
                if study_score is not None:  
                    total_score = study_score + bonus_malus_delta_real
                else:
                    total_score = study_score

        if classification_study_design == 1:  # Single Arm
            total_score_cap_single_arm = "⚠️ Total score capped at 3 for Single Arm studies."
            match, response_analysis = run_single_arm_criteria(text)
            if match == 1:
                study_score = 3
            elif match == 2:
                study_score = 2
            elif match == 3:
                study_score = 1
            else:
                study_score = 0
            if study_score >= 1:
                rules_data = get_bonus_and_malus_rules(text)
                bonus_malus_table_new = render_bonus_malus_rules_table(
                    text, rules_data)
                bonus_malus_delta = calculate_bonus_malus_delta(rules_data)
                print(bonus_malus_delta)
                if bonus_malus_delta >= 1:
                    bonus_malus_delta_real = 1
                elif bonus_malus_delta <= -1:
                    bonus_malus_delta_real = -1
                if study_score is not None:  
                    total_score = study_score + bonus_malus_delta_real
                else:
                    total_score = study_score
                if total_score > 3:
                    total_score = 3

        elif classification_study_design == 2:  # Case Report
            total_score = 0
            flag = "⚠️ Please review that case manually."

        elif classification_study_design == 3:  # Other
            total_score = None
            flag = "⚠️ This is not an RCT, Single Arm study, or Case Report. Please review manually."

    # Map score to letter
    if total_score is not None:
        if total_score >= 4:
            score_letter = "A"
        elif total_score == 3:
            score_letter = "B"
        elif total_score == 2:
            score_letter = "C"
        elif total_score == 1:
            score_letter = "C"
        else:
            score_letter = "D"

    return {
        "classification": classification_onc,
        "explanation": explanation,
        "metadata": metadata,
        "study_design": classification_study_design,
        "study_design_explanation": study_design_explanation,
        "total_score": total_score,
        "score_letter": score_letter,
        "response_analysis": response_analysis,
        "flag": flag,
        "rct_type": rct_type,
        "rct_type_explanation": rct_type_explanation,
        "study_score_explanation": study_score_explanation,
        "study_score": study_score,
        "bonus_malus_delta": bonus_malus_delta,
        "bonus_malus_delta_real": bonus_malus_delta_real,
        "total_score_cap_single_arm": total_score_cap_single_arm,
        "overarching_rules_table": overarching_rules_table_html,
        "match_score": match,
        "bonus_malus_table_new": bonus_malus_table_new
    }


def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


# ONCOLOGY CLASSIFICATION
def run_oncology_classification_prompt(text):
    rule = "Is the study oncological (cancer-related) or non-oncological (everything else)?"
    onc_prompt = build_onc_prompt(rule, text)

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": onc_prompt
        }],
        temperature=0.2,
    )

    raw_response = classification_response.choices[0].message.content.strip()
    print("🧠 Raw GPT Output:", raw_response)

    # 🧼 Remove ```json ... ``` wrapper if present
    if raw_response.startswith("```json"):
        raw_response = raw_response.strip("```json").strip("```").strip()

    try:
        parsed = json.loads(raw_response)
        return parsed["classification"], parsed["explanation"]
    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        return None, raw_response


def build_onc_prompt(rule_onc_text, text=""):
    return f"""You are a Swiss reimbursement reviewer.

Your task: Answer the following question: {rule_onc_text}

Study excerpt:
{text[:15000]}

Return your answer in the following JSON format:

{{
  "classification": 0 or 1,
  "explanation": "The text that you base your answer on."
}}

- Use 0 if the study is oncological (cancer-related)
- Use 1 if the study is non-oncological (everything else)
- Keep the structure exactly as shown
"""


# METADATA CLASSIFICATION
def run_metadata_classification_prompt(text):
    rule = "Provide the following metadata: Title of Study, Title of Journal and the following information of the study participants: Age, Gender, Ethnicity, Previous therapies. If you don't find some of the information, tell us."
    meta_prompt = build_meta_prompt(rule, text)

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": meta_prompt
        }],
        temperature=0.2,
    )

    raw_output = classification_response.choices[0].message.content.strip()

    # Remove common leading phrases like "Based on the provided excerpt..." or similar
    cleaned_output = re.sub(
        r"(?i)^based on.*?\n+",  # Case-insensitive match, remove leading line
        "",
        raw_output).strip()

    return cleaned_output


def build_meta_prompt(rule_onc_text, text=""):
    return f"""You are a Swiss reimbursement reviewer.
Your task: Answer the following question: {rule_onc_text}

Study excerpt:
{text[:30000]} 

"""


# STUDY DESIGN CLASSIFICATION
def run_study_design_classification_prompt(text):
    rule = "What is the design of the study?"
    study_design_prompt = build_study_design_prompt(rule, text)

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": study_design_prompt
        }],
        temperature=0.2,
    )

    raw_response = classification_response.choices[0].message.content.strip()
    print("🧠 Raw GPT Output (Study Design):", raw_response)

    if raw_response.startswith("```json"):
        raw_response = raw_response.strip("```json").strip("```").strip()

    try:
        parsed = json.loads(raw_response)
        return parsed["classification"], parsed["explanation"]
    except Exception as e:
        print("⚠️ JSON parsing failed (Study Design):", e)
        return None, raw_response


def build_study_design_prompt(rule, text):
    return f"""You are a Swiss reimbursement reviewer.

Your task: Answer the following question: {rule}

Study excerpt:
{text[:15000]}

Return your answer in the following JSON format:

{{
  "classification": 0, 1, 2 or 3,
  "explanation": "The text that you base your answer on."
}}

- Use 0 if the study is a Randomized Control Trial (RCT)
- Use 1 if the study is a Single Arm study
- Use 2 if it's a Case Report
- Use 3 for Other
- Keep the structure exactly as shown
"""


# If classification_onc == 1 & classification_study_design == 0: RCT
def get_rct_type(text):
    type_prompt = """
You are a medical analyst.

Your task is to determine the **study type** based on the *primary endpoint* and *how it is measured*.

The 6 possible study types are:

1. **Funktionsänderung (Functional Change)** – Measured by % change from baseline  
2. **Anzahl Ereignisse (Event Count)** – Measured by % change from baseline  
3. **Scorepoints** – Measured by % change from baseline  
4. **Surrogatwerte mit Studienrange (x–y)** – Measured by % change from baseline (e.g. biomarker levels)  
5. **Ansprechrate eines Ziels im 1° Endpunkt** – Measured by % response rate  
6. **Mortalität** – Measured by hazard ratio (HR) or % mortality reduction over time

Return your answer in this JSON format:
{
  "type": 1 to 6,
  "explanation": "Text excerpt justifying the classification"
}
"""

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role":
            "user",
            "content":
            type_prompt + "\n\nStudy excerpt:\n" + text[:15000]
        }],
        temperature=0.2)

    raw = response.choices[0].message.content.strip()

    # Extract JSON safely
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return parsed.get("type"), parsed.get("explanation")
        except Exception as e:
            print("❌ Error parsing RCT type JSON:", e)
            return None, raw
    else:
        print("❌ RCT type JSON block not found")
        return None, raw


def get_rct_category_type_1_to_4(text):
    prompt = """
You are a medical evaluator.

Your task is to assign a numerical score (study_score) based on the **magnitude of clinical benefit** reported for the **primary endpoint** of the study. Your top priority is to find an explicitly stated **Δ% (percentage difference)**, between the treatment (verum) and control group for the **primary endpoint**.

- Do **not** use hazard ratios unless no Δ% is reported anywhere (or can be calculated based on the information provided) for the primary endpoint.
- Only use relapse rate or risk measures if the **relative reduction (Δ%)** is stated in the **same sentence or nearby context**.

---

Scoring rules:

If Δ% is reported for the primary endpoint:
- Δ% ≥ 50% → study_score = 4
- 30% ≤ Δ% < 50% → study_score = 3
- 10% ≤ Δ% < 30% → study_score = 2
- 5% ≤ Δ% < 10% → study_score = 1

If no Δ% is reported, use hazard ratio for the primary endpoint:
- HR ≤ 0.65 → study_score = 3
- HR ≤ 0.70 → study_score = 2
- HR ≤ 0.75 → study_score = 1

   
Return only this JSON:
{
  "study_score": 1 to 4,
  "explanation": "Return the information that you used to determine the score (i.e. Δ% or HR)."
}
"""

    return _run_rct_category_prompt(prompt, text)


def get_rct_category_type_5(text):
    prompt = """
You are a medical evaluator.

This study is measured by **% response rate to the primary endpoint** (=Δ%).  
Use the following rules to determine the **study_score** based on the response rate that the study mentions. Use the first category that matches:

If Δ% ≥ 40%, then study_score = 4.
If 20% ≤ Δ% < 40%, then study_score = 3.
If 10% ≤ Δ% < 20%, then study_score = 2.
If 5% ≤ Δ% < 10%, then study_score = 1.

Return only this JSON:
{
  "study_score": 1 to 4,
  "explanation": "Text excerpt justifying the category"
}
"""

    return _run_rct_category_prompt(prompt, text)


def get_rct_category_type_6(text):
    prompt = """
You are a medical evaluator.

This study is evaluated by **mortality reduction per year** (ΔOS) or **hazard ratio (HR)**. Either take the ΔOS or HR that the study mentions or calculate them.

Based on the values use the following rules to determine the **study_score**. Use the first category that matches:

If ΔOS ≥ 2% per year, then study_score = 4.
If 2% ≥ ΔOS ≥ 1% per year OR HR ≤ 0.80, then study_score = 3.
If 1% ≥ ΔOS ≥ 0.5% per year OR HR ≤ 0.85, then study_score = 2.
If ΔOS < 0.5% per year OR HR > 0.85, then study_score = 1.

Return only this JSON:
{
  "study_score": 1 to 4,
  "explanation": "Text excerpt justifying the category"
}
"""

    return _run_rct_category_prompt(prompt, text)


def _run_rct_category_prompt(prompt, text):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": prompt + "\n\nStudy excerpt:\n" +
            text[:150000]  #check, might be too long
        }],
        temperature=0.2)

    raw = response.choices[0].message.content.strip()
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return parsed.get("study_score"), parsed.get("explanation")
        except Exception as e:
            print("❌ Error parsing RCT category JSON:", e)
            #return None, raw
            return 0, raw
    else:
        print("❌ RCT category JSON block not found")
        #return None, raw
        return 0, raw


def get_rct_category(text, rct_type):
    if rct_type in [1, 2, 3, 4]:
        return get_rct_category_type_1_to_4(text)
    elif rct_type == 5:
        return get_rct_category_type_5(text)
    elif rct_type == 6:
        return get_rct_category_type_6(text)
    else:
        print("❌ Unknown RCT type")
        return None, "RCT type not recognized"


# If classification_onc == 1 & classification_study_design == 1: Single Arm study
def run_single_arm_criteria(text):
    rule = """Check the following:
1) Is the Ansprechen (Response Rate) > 60% **and** Duration of Response (DoR) > 6 Monate?
2) Is the Ansprechen (Response Rate) ≥ 30%?
3) Is there no information about response rate?

Only return the first matching option. If none apply (e.g., response rate < 30%), return that.

Return result in this JSON format:
{
  "match": 1, 2, 3, or 0,
  "explanation": "Relevant excerpt from the study."
}
"""
    prompt = f"""You are a Swiss reimbursement reviewer.

{rule}

Study excerpt:
{text[:15000]}
"""

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,
    )

    raw_response = classification_response.choices[0].message.content.strip()
    print("📊 Raw GPT Output (Single Arm Evaluation):", raw_response)

    try:
        if raw_response.startswith("```json"):
            raw_response = raw_response.strip("```json").strip("```").strip()

        parsed = json.loads(raw_response)
        return parsed["match"], parsed["explanation"]
    except Exception as e:
        print("⚠️ JSON parsing failed (Single Arm Criteria):", e)
        return 0, "Unable to parse response."


# BONUS AND MALUS FUNCTION
def get_bonus_and_malus_rules(text):
    rules_with_prompts = [{
        "display":
        """QoL: Deklariert als 2° Endpunkt, p ≤ 0,05 und mit validierten Fragebogen erhoben. Kein Bonus bei «non inferior» oder falls Bonus Teil von 1° Endpunkt.""",
        "prompt":
        """Is Quality of Life (QoL) measured in the study? 
            If QoL is declared as a secondary endpoint, p ≤ 0.05, and measured with validated questionnaires, return "+1".
            If QoL is part of the primary endpoint, do not return "+1".
            If the study is a non-inferiority study, do not return "+1"
            """
    }, {
        "display":
        "sAE (Serious Adverse Events): ∆ sAE ≥ 35% (∆ = Verum ↔ Kontrollgruppe) oder ≥ 50% bei single-arm ",
        "prompt":
        """Are serious adverse events reported? If the study is a randomized controlled trial and the difference in serious adverse events between the treatment and control group is ≥ 35%, return "-1". If the study is a single-arm study and serious adverse events occur at ≥ 50%, return "-1"."""
    }, {
        "display":
        "Ansprechen (RR = Response Rate): RR 15-30%. Nicht bei Typ 5 und single-arm. (RR < 15 % = Max. Studienrating C)",
        "prompt":
        """Do not apply this rule for single arm studies or type 5 randomized control trial with response rate as the primary endpoint: If the Response rate (RR) is between 15% and 30%, return "-1" """
    }, {
        "display":
        """Studiendefizite: Kontrollgruppe nicht adäquat dargestellt oder historische Kontrollgruppe
mit unpräzis definierten Kriterien, Studienvolltext liegt nicht vor → Die Studie ist damit nicht voll beurteilbar, Langzeitwirkung ist aus Studie nicht plausibel ableitbar, Andere klinisch relevante Defizite, welche der VA begründen kann.""",
        "prompt":
        """Do any one of these major study deficits exist?
                inadequate control group or imprecise historical control, 
                full-text study not available, 
                long-term effects not plausibly derivable, 
                other clinically relevant deficits determined by VA.
            If yes, return "-1". If no, return "Non-applicable"."""
    }, {
        "display":
        """Expertenbeizug: Zur Klärung einer für den VA unklaren klinischen Relevanz oder zur Klärung von Besonderheiten der Studie """,
        "prompt": None
    }]

    results = []
    for i, r in enumerate(rules_with_prompts):
        if r["prompt"] is None:
            results.append({
                "rule": r["display"],
                "response": {
                    "suggested_flag": "Non-applicable",
                    "excerpt": "Non-applicable"
                }
            })
        else:
            results.append({
                "rule":
                r["display"],
                "response":
                get_chatgpt_response_bonus_malus(r["prompt"], text)
            })
    return results


def get_chatgpt_response_bonus_malus(rule, text):
    import json
    import re

    prompt = f"""You are a Swiss reimbursement reviewer. Answer the following question:

{rule}

Based on the study excerpt:
{text[:30000]}

Return your answer in the following JSON format:

```json
{{
  "suggested_flag": "+1, 0, or -1 as appropriate",
  "excerpt": "Text excerpt or reasoning used to justify your decision."
}}
```"""

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,
    )

    raw_response = classification_response.choices[0].message.content.strip()
    print("📊 Raw GPT Output:", raw_response)

    try:
        json_match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL)
        if json_match:
            raw_response = json_match.group(1).strip()

        parsed = json.loads(raw_response)
        return {
            "suggested_flag": parsed.get("suggested_flag", "⚠️ No flag"),
            "excerpt": parsed.get("excerpt", "⚠️ No excerpt provided")
        }
    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        return {
            "suggested_flag": "⚠️ Review required",
            "excerpt": "⚠️ Parsing failed – apply expert judgment"
        }


def render_bonus_malus_rules_table(text, rules_data):
    html = [
        #'<h3>📊 Bonus-Malus Bewertung</h3>',
        '<table border="1" style="width:100%; border-collapse: collapse;">',
        '<thead>',
        '<tr><th>Regel</th><th>Empfehlung</th><th>Begründung (Ausschnitt)</th></tr>',
        '</thead>',
        '<tbody>'
    ]

    for row in rules_data:
        html.append(f"<tr>"
                    f"<td>{row['rule']}</td>"
                    f"<td>{row['response']['suggested_flag']}</td>"
                    f"<td>{row['response']['excerpt']}</td>"
                    f"</tr>")

    html.append('</tbody></table>')
    return '\n'.join(html)


def calculate_bonus_malus_delta(rules_data):
    total = 0
    for row in rules_data:
        flag = row['response'].get('suggested_flag', '0')
        try:
            # Only sum if it's actually a valid number (+1, 0, -1)
            flag_int = int(flag)
            total += flag_int
        except ValueError:
            # If it's "Non-applicable" or parsing fails, just skip it
            continue
    return total


# OVERARCHING RULES TABLE
def get_overarching_rules(text):
    rules_with_prompts = [{
        "display":
        "Suggest a separate subgroup rating if a subgroup is sufficiently powered and statistically significant",
        "prompt":
        "Is there a statistically significant (p ≤ 0,05) and sufficiently powered subgroup that would justify a separate rating? If yes, provide proof from the study & suggest to run a separate subgroup rating. Output the response in German"
    }, {
        "display":
        "Discrepancies between primary endpoint and reported result or missing/unclear baseline data result in a maximum study rating of C",
        "prompt":
        "Are there discrepancies between primary endpoint and reported result or missing/unclear baseline data? Explain what you have found in the study text. Do not suggest a rating. Output the response in German"
    }, {
        "display":
        "Studies without clear indication, \"nur Zulassung auf ein genetisches Merkmal\", receive a maximum study rating of C",
        "prompt":
        "Is this medication without clear indication and only geared towards genetic material? Explain what you have found in the study text but do not suggest a rating. Output the response in German"
    }, {
        "display":
        "Studies based only on surrogate endpoints or biological markers are capped at overall rating B, determined by case-by-case VA review",
        "prompt":
        "Is this study based only on surrogate endpoints or biological markers? If yes, return text from the study to prove your judgement, flag for case-by-case review and cap overall rating at B. Output the response in German"
    }, {
        "display":
        "Response rate (RR) < 15% (excluding Type 5 and single-arm) results in a maximum study rating of C",
        "prompt":
        "Do not apply for single arm studies or type 5 randomized control trial with response rate as the primary endpoint: Is the Response rate (RR) < 15%? If yes, return text from the study to prove your judgement & cap maximum study rating of C. Output the response in German"
    }]

    return [{
        "rule": r["display"],
        "note": "Manual flag required",
        "suggested_flag": get_chatgpt_response(r["prompt"], text)
    } for r in rules_with_prompts]


def get_chatgpt_response(rule, text):
    import json
    import re

    prompt = f"""You are a Swiss reimbursement reviewer. Answer the following question:

{rule}

Based on the study excerpt:
{text[:30000]}
Return your answer in the following JSON format:

```json
{{
  "suggested_flag": "Your short recommendation here"
}}
```"""

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,
    )

    raw_response = classification_response.choices[0].message.content.strip()
    print("📊 Raw GPT Output (Overarching Rule):", raw_response)

    try:
        # Clean and parse JSON from code block
        json_match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL)
        if json_match:
            raw_response = json_match.group(1).strip()

        parsed = json.loads(raw_response)
        return parsed.get("suggested_flag", "⚠️ No flag returned")
    except Exception as e:
        print("⚠️ JSON parsing failed (Overarching Rule):", e)
        return "⚠️ Review required – apply expert judgment"


def render_overarching_rules_table(text):
    rules_data = get_overarching_rules(text)
    html = [
        #'<h3>Overarching Rules</h3>',
        '<table border="1" style="width:100%; border-collapse: collapse;">'
    ]
    html.append(
        "<thead><tr><th>Overarching Rule</th><th>Manual Note</th><th>Suggested Flag</th></tr></thead><tbody>"
    )

    for row in rules_data:
        html.append(
            f"<tr><td>{row['rule']}</td><td>{row['note']}</td><td>{row['suggested_flag']}</td></tr>"
        )

    html.append("</tbody></table>")
    return '\n'.join(html)


###################
# OLD CODE
def bonus_malus_func(study_score, text):
    import re
    import json

    rule = """
    You are a Swiss reimbursement reviewer.

    Please evaluate the following study excerpt based on specific bonus/malus criteria used in Swiss reimbursement assessments.

    Answer the following 5 questions and provide your assessment in a tabular format with 4 columns:

    1. **Kriterium** – The name of the criterion (e.g., QoL, Serious AE, etc.)
    2. **Regel** – The rule in German that defines the criterion (see below)
    3. **Bewertung** – One of:
       - "+1" if the criterion clearly applies positively
       - "-1" if the criterion clearly applies negatively
       - "non-applicable" if the criterion does not apply or cannot be assessed
       Bewertung for each criteria is written below in brackets after the rule.
    4. **Begründung** – Relevant excerpt from the study.

    Return:
    - A **Markdown-style table** with the above columns (5 rows total). For the fifth row, use "Manual flag required" as Bewertung, and leave the Begründung cell empty.
    - A final field `"adjustment"` that is the **sum** of all numeric Bewertungen (i.e. consider the Bewertung only if the value is either +1 or -1), excluding "non-applicable".

    ### Bewertung criteria:

    - **QoL**: Deklariert als 2°, p ≤ 0,05 und mit validierten Fragebogen erhoben. Kein Bonus bei «non inferior» oder falls Bonus Teil von 1° Endpunkt. *(+1)*
    - **Serious Adverse Events (sAE)**: ΔsAE (= Differenz zwischen Verum und Kontrollgruppe) ≥ 35% bei RCT oder ΔsAE ≥ 50% bei Single-Arm-Studien. *(-1)*
    - **Ansprechen (also called response rate = RR)**: RR 15–30 %. Nicht bei Typ-5-Indikation oder Single-Arm-Studien. (RR < 15 % = Max. Studienrating C). *(-1)*
    - **Studiendefizite**: Kontrollgruppe nicht adäquat dargestellt, historische Kontrollgruppe mit unpräzisen Kriterien, fehlender Volltext, unplausible Langzeitwirkung oder andere relevante Mängel. *(-1)*
    - **Expertenbeizug durch VA**: Bei unklarer klinischer Relevanz oder besonderen Studienmerkmalen. *(manual flag required)*

    ### Return format (JSON):
    ```json
    {
      "adjustment": sum of all Bewertungen,
      "table": "Markdown-style table with Kriterium | Regel | Bewertung | Begründung"
    }
    """

    prompt = f"""You are a Swiss reimbursement reviewer.

{rule}

Study excerpt:
{text[:30000]}
"""

    classification_response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,
    )

    raw_response = classification_response.choices[0].message.content.strip()
    print("📋 Raw GPT Output (Bonus/Malus):", raw_response)

    try:
        # Extract JSON code block content using regex
        json_match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL)
        if json_match:
            raw_response = json_match.group(1).strip()

        parsed = json.loads(raw_response)
        return parsed["adjustment"], parsed["table"]

    except Exception as e:
        print("⚠️ JSON parsing failed (Bonus/Malus):", e)
        return 0, f"⚠️ Unable to parse bonus/malus response.\n\nRaw:\n{raw_response}"


# old
def render_bonus_malus_table(bonus_malus_table):
    """
    Converts a Markdown-style table from GPT response into HTML format.
    """
    import re

    lines = bonus_malus_table.strip().split('\n')

    # Only keep rows that look like a markdown table line (with | separators)
    rows = [
        line for line in lines
        if '|' in line and not re.match(r'^\s*\|?\s*-+\s*\|', line)
    ]

    if not rows:
        return "<p>⚠️ Keine gültige Bonus-Malus Tabelle gefunden.</p>"

    html = [
        '<h3>Bonus-Malus Bewertung</h3>',
        '<table border="1" style="width:100%; border-collapse: collapse;">',
        #'<thead><tr><th>Kriterium</th><th>Regel</th><th>Bewertung</th><th>Begründung</th></tr></thead>',
        '<tbody>'
    ]

    for row in rows:
        # Split markdown table row into cells
        cells = [cell.strip() for cell in row.strip('|').split('|')]
        if len(cells) == 4:
            html.append('<tr>' + ''.join(f'<td>{cell}</td>'
                                         for cell in cells) + '</tr>')

    html.append('</tbody></table>')

    return '\n'.join(html)
