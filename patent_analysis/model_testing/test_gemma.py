from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# files
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"

model_name = "google/gemma-3-4b-it"
input_file = data_dir / "patent_validation_50_labeled.csv"
output_file = data_dir / "gemma_validation_results.csv"

# load validation data
df = pd.read_csv(input_file, sep=";")

# load model
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

model.eval()

# screening prompt
system_prompt = """
You are screening patents for relevance to anion exchange membrane
water electrolysis (AEMWE).

Classify each patent as exactly one of:

Include
Exclude

Include only if the title or abstract explicitly connects the invention to:
- anion exchange membrane water electrolysis,
- AEMWE,
- hydrogen production by water electrolysis using an anion exchange membrane,
- or a component explicitly intended for such a system.

Exclude if:
- the anion exchange membrane is used for another electrochemical application,
- water electrolysis is generic and no explicit AEM connection is established,
- the invention concerns fuel cells, CO2 electrolysis, salt electrolysis,
  electrodialysis, wastewater treatment, chemical production,
  microbial electrolysis, metal recovery, or another non-AEMWE application.

Important:
- Do not infer relevance from possible use.
- "anion exchange membrane" alone is not enough.
- "water electrolysis" alone is not enough.
- Both the AEM connection and the water-electrolysis/hydrogen-production
  connection must be explicit.
- If explicit evidence is missing, choose Exclude.

Return exactly:

Decision: Include or Exclude
Reason: one short sentence
"""

gemma_decisions = []
gemma_reasons = []

# screen patents
for index, row in df.iterrows():

    title = str(row.get("Title", ""))
    abstract = str(row.get("Abstract", ""))

    user_prompt = f"""Title:
{title}

Abstract:
{abstract}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # format prompt
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    # run model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    decision = ""
    reason = ""

    # read output
    for line in text.splitlines():
        line = line.strip()

        if line.lower().startswith("decision:"):
            decision = line.split(":", 1)[1].strip()

        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    # clean decision
    if decision.lower() == "include":
        decision = "Include"

    elif decision.lower() == "exclude":
        decision = "Exclude"

    else:
        decision = "Parse Error"

    gemma_decisions.append(decision)
    gemma_reasons.append(reason)

    print(f"{index + 1}/{len(df)} | {decision} | {title[:60]}")

# add predictions
df["Gemma Decision"] = gemma_decisions
df["Gemma Reason"] = gemma_reasons

# clean labels
df["Human Decision Clean"] = (
    df["Human Decision"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)

df["Gemma Decision Clean"] = (
    df["Gemma Decision"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)

# keep valid predictions
valid_df = df[
    df["Gemma Decision Clean"].isin(["Include", "Exclude"])
].copy()

# confusion matrix values
tp = len(valid_df[
    (valid_df["Human Decision Clean"] == "Include")
    & (valid_df["Gemma Decision Clean"] == "Include")
])

tn = len(valid_df[
    (valid_df["Human Decision Clean"] == "Exclude")
    & (valid_df["Gemma Decision Clean"] == "Exclude")
])

fp = len(valid_df[
    (valid_df["Human Decision Clean"] == "Exclude")
    & (valid_df["Gemma Decision Clean"] == "Include")
])

fn = len(valid_df[
    (valid_df["Human Decision Clean"] == "Include")
    & (valid_df["Gemma Decision Clean"] == "Exclude")
])

# metrics
total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0
)

# results
print("\nGEMMA VALIDATION RESULTS")

print("\nConfusion matrix:")
print(pd.crosstab(
    valid_df["Human Decision Clean"],
    valid_df["Gemma Decision Clean"]
))

print("\nTP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

print("\nAccuracy:", round(accuracy * 100, 1), "%")
print("Precision:", round(precision * 100, 1), "%")
print("Recall:", round(recall * 100, 1), "%")
print("Specificity:", round(specificity * 100, 1), "%")
print("F1-score:", round(f1 * 100, 1), "%")

# save results
df.to_csv(output_file, index=False)