import pandas as pd

# files
input_file = "data/lens_broader_export.csv"
output_file = "data/patent_validation_50.csv"

# load data
df = pd.read_csv(input_file)

print("Total patents:", len(df))

# remove duplicate patents
df = df.drop_duplicates(subset="Lens ID")

# create random sample
sample = df.sample(
    n=50,
    random_state=42
).copy()

# keep useful columns
wanted_columns = [
    "Lens ID",
    "Title",
    "Abstract",
    "Applicants",
    "Earliest Priority Date",
    "CPC Classifications",
    "IPCR Classifications",
    "URL"
]

available_columns = [
    column for column in wanted_columns
    if column in sample.columns
]

sample = sample[available_columns]

# add manual labels
sample["Human Decision"] = ""
sample["Human Reason"] = ""

# save
sample.to_csv(
    output_file,
    index=False
)