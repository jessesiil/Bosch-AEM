import pandas as pd
from pathlib import Path


# files
project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "qwen_validation_results.csv"
output_file = project_root / "data" / "aem_patents_final.csv"

# load data
df = pd.read_csv(input_file)

print("Rows loaded:", len(df))

# keep relevant patents
df = df[
    df["Qwen Decision"] == "Include"
].copy()

print("Qwen includes:", len(df))

# create priority year
df["Earliest Priority Date"] = pd.to_datetime(
    df["Earliest Priority Date"],
    errors="coerce"
)

df["Priority Year"] = df["Earliest Priority Date"].dt.year

# keep patents from 2010 onwards
df = df[
    df["Priority Year"] >= 2010
].copy()

# remove duplicate patents
before = len(df)

df = df.drop_duplicates(
    subset=["Lens ID"]
).copy()

print("Duplicate Lens IDs removed:", before - len(df))

# check important columns
important_columns = [
    "Title",
    "Abstract",
    "Applicants",
    "Earliest Priority Date",
    "Jurisdiction"
]

print("\nMissing values:")

for column in important_columns:
    if column in df.columns:
        print(column, ":", df[column].isna().sum())

# check EP families
family_column = "Simple Family Member Jurisdictions"

ep_families = df[
    df[family_column]
    .fillna("")
    .str.contains(r"\bEP\b", regex=True)
]

print("\nFinal patents:", len(df))
print("EP-containing families:", len(ep_families))

# save
df.to_csv(output_file, index=False)