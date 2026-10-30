import os
from pathlib import Path

import pandas as pd
from ollama import chat

# files
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"

input_file = data_dir / "lens_narrower_export.csv"
output_file = data_dir / "lens_refined_qwen_screened.csv"

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

# load existing results if available
if os.path.exists(output_file):
    df = pd.read_csv(output_file)
    print("Existing results found. Resuming...")
else:
    df = pd.read_csv(input_file)
    df["Qwen Decision"] = ""
    df["Qwen Reason"] = ""

print("Patents loaded:", len(df))

# screen patents
for index, row in df.iterrows():

    # skip finished rows
    if str(row["Qwen Decision"]).strip() in {"Include", "Exclude"}:
        continue

    title = str(row.get("Title", ""))
    abstract = str(row.get("Abstract", ""))

    user_prompt = f"""Title:
{title}

Abstract:
{abstract}
"""

    try:

        response = chat(
            model="qwen3:8b",
            think=False,
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        decision = ""
        reason = ""

        # read model output
        for line in response.message.content.strip().splitlines():
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

        # save result
        df.at[index, "Qwen Decision"] = decision
        df.at[index, "Qwen Reason"] = reason

        print(f"{index + 1}/{len(df)} | {decision} | {title[:60]}")

    except Exception as e:

        print(f"ERROR on patent {index + 1}: {e}")
        df.at[index, "Qwen Decision"] = "Error"

    # save progress
    if (index + 1) % 10 == 0:
        df.to_csv(output_file, index=False)

# final save
df.to_csv(output_file, index=False)

print("\nSaved:", output_file)