import pandas as pd

# files
refined_file = "data/lens_narrower_export.csv"
validation_file = "data/patent_validation_50_labeled.csv"
output_file = "data/refined_query_validation_results.csv"

# load data
refined = pd.read_csv(refined_file)

validation = pd.read_csv(
    validation_file,
    sep=";"
)

print("Refined dataset size:", len(refined))
print("Validation sample size:", len(validation))

# remove duplicate patents
refined = refined.drop_duplicates(subset="Lens ID")
validation = validation.drop_duplicates(subset="Lens ID")

# check which validation patents remain in refined query
validation["Found in Refined Query"] = validation["Lens ID"].isin(
    refined["Lens ID"]
)

# summarize by manual label
summary = (
    validation
    .groupby("Human Decision")["Found in Refined Query"]
    .agg(["sum", "count"])
)

summary["Percentage Found"] = (
    summary["sum"] / summary["count"] * 100
).round(1)

print("\nResults by human decision:")
print(summary)

# save
validation.to_csv(
    output_file,
    index=False
)