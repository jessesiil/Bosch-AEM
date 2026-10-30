import re
from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory


# files
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "aem_patents_final.csv"
output_file = project_root / "data" / "aem_patents_cleaned.csv"

# load data
df = pd.read_csv(input_file)

print("Rows loaded:", len(df))

# check columns
required_columns = ["Lens ID", "Title", "Abstract"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# fill missing text
df["Title"] = df["Title"].fillna("")
df["Abstract"] = df["Abstract"].fillna("")

# remove patents without title or abstract
before = len(df)

df = df[
    (df["Title"].str.strip() != "")
    | (df["Abstract"].str.strip() != "")
].copy()

print("Rows removed because of missing text:", before - len(df))

# detect abstract language
DetectorFactory.seed = 42

def detect_language(text):
    try:
        return detect(str(text))
    except Exception:
        return "unknown"

df["Abstract Language"] = df["Abstract"].apply(detect_language)

print("\nAbstract language distribution:")
print(df["Abstract Language"].value_counts())

# keep English patents
before = len(df)

df = df[
    df["Abstract Language"] == "en"
].copy()

print("Non-English patents removed:", before - len(df))

# combine title and abstract
df["Text Raw"] = (
    df["Title"].str.strip()
    + " "
    + df["Abstract"].str.strip()
)

# clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Text Clean"] = df["Text Raw"].apply(clean_text)

# create priority year
if "Earliest Priority Date" in df.columns:
    df["Priority Year"] = pd.to_datetime(
        df["Earliest Priority Date"],
        errors="coerce"
    ).dt.year

# remove duplicate patents
before = len(df)

df = df.drop_duplicates(
    subset=["Lens ID"]
).copy()

print("Duplicate Lens IDs removed:", before - len(df))

# final checks
print("Final rows:", len(df))
print("Missing cleaned text:", df["Text Clean"].isna().sum())
print(
    "Empty cleaned text:",
    (df["Text Clean"].str.strip() == "").sum()
)

# save
df.to_csv(output_file, index=False)