import pandas as pd
from pathlib import Path
import numpy as np


# files

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "outputs" / "tables" / "aem_topics_6.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "tables" / "aem_topic_overlap.csv"


# load data

df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))


# topic score columns

topic_columns = [
    "Topic 1 Score",
    "Topic 2 Score",
    "Topic 3 Score",
    "Topic 4 Score",
    "Topic 5 Score",
    "Topic 6 Score"
]

for column in topic_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}"
        )


# calculate top and second topic

topic_scores = df[topic_columns].to_numpy()

sorted_indices = np.argsort(
    topic_scores,
    axis=1
)

top_indices = sorted_indices[:, -1]
second_indices = sorted_indices[:, -2]


df["Top Topic"] = top_indices + 1

df["Second Topic"] = second_indices + 1


df["Top Score"] = topic_scores[
    np.arange(len(df)),
    top_indices
]

df["Second Score"] = topic_scores[
    np.arange(len(df)),
    second_indices
]


# calculate difference between top two scores

df["Score Difference"] = (
    df["Top Score"]
    - df["Second Score"]
)


# summary statistics

print("\nScore difference summary:")

print(
    df["Score Difference"]
    .describe()
)


# test possible thresholds

thresholds = [
    0.02,
    0.05,
    0.10
]

print("\nPossible mixed-topic thresholds:")

for threshold in thresholds:

    mixed_count = (
        df["Score Difference"] < threshold
    ).sum()

    mixed_percentage = (
        mixed_count / len(df)
    ) * 100

    print(
        f"Below {threshold:.2f}: "
        f"{mixed_count} patents "
        f"({mixed_percentage:.1f}%)"
    )


# create temporary mixed label using 0.05

MIXED_THRESHOLD = 0.05

df["Topic Assignment"] = np.where(
    df["Score Difference"] < MIXED_THRESHOLD,
    "Mixed",
    "Clear"
)


# summary of clear and mixed patents

print("\nTopic assignment:")

print(
    df["Topic Assignment"]
    .value_counts()
)

print("\nTopic assignment percentage:")

print(
    df["Topic Assignment"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(1)
)


# most common topic combinations among mixed patents

mixed_df = df[
    df["Topic Assignment"] == "Mixed"
].copy()

mixed_df["Topic Pair"] = (
    "Topic "
    + mixed_df["Top Topic"].astype(str)
    + " + Topic "
    + mixed_df["Second Topic"].astype(str)
)

print("\nMost common mixed topic pairs:")

print(
    mixed_df["Topic Pair"]
    .value_counts()
    .head(15)
)


# save useful columns

columns_to_save = [
    "Lens ID",
    "Title",
    "Abstract",
    "Priority Year",
    "Applicants",
    "Top Topic",
    "Top Score",
    "Second Topic",
    "Second Score",
    "Score Difference",
    "Topic Assignment",
    "Topic 1 Score",
    "Topic 2 Score",
    "Topic 3 Score",
    "Topic 4 Score",
    "Topic 5 Score",
    "Topic 6 Score"
]

result_df = df[
    columns_to_save
].copy()


# save

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)