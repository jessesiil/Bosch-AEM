import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# files
project_root = Path(__file__).resolve().parent

data_dir = project_root / "data"
table_dir = project_root / "outputs" / "tables"
figure_dir = project_root / "outputs" / "figures"

table_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

topic_file = table_dir / "aem_topics_6.csv"
patent_file = data_dir / "aem_patents_final.csv"

technical_review_file = table_dir / "topic_2_3_technical_review.csv"
ep_topic_file = table_dir / "ep_topic_analysis.csv"

patent_trend_figure = figure_dir / "aem_patents_by_year.png"
topic_distribution_figure = figure_dir / "patent_topic_distribution.png"
topic_share_figure = figure_dir / "topic_2_3_share_by_year.png"
ep_figure = figure_dir / "ep_patents_by_topic.png"


# settings
start_year = 2010
end_year = 2025

topic_names = {
    1: "MEA and catalyst-layer architecture",
    2: "AEM polymer chemistry and membrane stability",
    3: "OER catalysts and catalytic electrodes",
    4: "Hydrogen-production systems and control",
    5: "Catalyst/electrode engineering and integration",
    6: "Cell and stack architecture"
}

score_columns = [
    "Topic 1 Score",
    "Topic 2 Score",
    "Topic 3 Score",
    "Topic 4 Score",
    "Topic 5 Score",
    "Topic 6 Score"
]


# figure style
background = "#ffffff"
title_color = "#1f1f1f"
grey = "#52514e"
grid_color = "#d9d9d6"
blue = "#2a78d6"
orange = "#eb6834"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "figure.facecolor": background,
    "axes.facecolor": background,
    "axes.edgecolor": grey,
    "axes.labelcolor": grey,
    "xtick.color": grey,
    "ytick.color": grey
})


# helper functions
def check_columns(df, columns):
    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def add_top_topic(df):
    df = df.copy()

    df["Top Topic"] = (
        df[score_columns]
        .idxmax(axis=1)
        .str.extract(r"Topic (\d+)")[0]
        .astype(int)
    )

    return df


def find_categories(text, categories):
    text = str(text).lower()
    found = []

    for category, terms in categories.items():
        if any(term in text for term in terms):
            found.append(category)

    return "; ".join(found)


def style_axes(ax):
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(grey)
    ax.spines["bottom"].set_color(grey)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        labelsize=9,
        width=0.8,
        length=3,
        colors=grey
    )


# load data
topic_df = pd.read_csv(topic_file)
patent_df = pd.read_csv(patent_file)

check_columns(
    topic_df,
    [
        "Lens ID",
        "Title",
        "Abstract",
        "Priority Year"
    ] + score_columns
)

check_columns(
    patent_df,
    [
        "Lens ID",
        "Priority Year",
        "Simple Family Member Jurisdictions"
    ]
)

topic_df = add_top_topic(topic_df)

print("Topic-model patents:", len(topic_df))
print("Final patent families:", len(patent_df))


# patent activity by year
patent_df["Priority Year"] = pd.to_numeric(
    patent_df["Priority Year"],
    errors="coerce"
)

trend_df = patent_df[
    patent_df["Priority Year"].between(
        start_year,
        end_year
    )
].copy()

patents_per_year = (
    trend_df["Priority Year"]
    .value_counts()
    .reindex(
        range(start_year, end_year + 1),
        fill_value=0
    )
    .sort_index()
)

print("\nPatent families by year:")
print(patents_per_year)

fig, ax = plt.subplots(
    figsize=(7.6, 5.2),
    dpi=200
)

ax.plot(
    patents_per_year.index,
    patents_per_year.values,
    color=blue,
    linewidth=2,
    marker="o",
    markersize=4.5
)

ax.set_title(
    "AEM patent activity by priority year",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=title_color,
    pad=12
)

ax.set_ylabel("Patent families")
ax.set_xlabel("")

ax.set_ylim(0, 140)
ax.set_yticks(range(0, 141, 20))

ax.set_xticks([
    2011,
    2013,
    2015,
    2017,
    2019,
    2021,
    2023,
    2025
])

ax.grid(
    axis="y",
    color=grid_color,
    linewidth=0.8
)

style_axes(ax)

fig.text(
    0.085,
    0.025,
    f"Source: Lens patent analysis; n={len(trend_df)} patent families "
    "included in the 2010–2025 trend analysis. "
    "2026 excluded as incomplete; recent years may be affected by publication lag.",
    fontsize=7,
    color=grey
)

plt.subplots_adjust(
    left=0.11,
    right=0.985,
    top=0.88,
    bottom=0.16
)

plt.savefig(
    patent_trend_figure,
    dpi=200,
    facecolor="white",
    bbox_inches="tight"
)

plt.close()


# topic distribution
topic_counts = (
    topic_df["Top Topic"]
    .value_counts()
    .reindex(
        range(1, 7),
        fill_value=0
    )
)

topic_result = pd.DataFrame({
    "Topic": range(1, 7),
    "Topic Name": [
        topic_names[topic]
        for topic in range(1, 7)
    ],
    "Patent Count": topic_counts.values
})

topic_result["Share (%)"] = (
    topic_result["Patent Count"]
    / len(topic_df)
    * 100
)

print("\nTopic distribution:")
print(topic_result.round(1))

plot_df = topic_result.sort_values(
    "Patent Count",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(7.6, 5.2),
    dpi=200
)

ax.barh(
    plot_df["Topic Name"],
    plot_df["Patent Count"],
    color=blue,
    height=0.62
)

ax.set_title(
    "AEM patent families by technical topic",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=title_color,
    pad=12
)

ax.set_xlabel("Patent families")
ax.set_ylabel("")

ax.set_xlim(0, 140)
ax.set_xticks(range(0, 141, 20))

ax.grid(
    axis="x",
    color=grid_color,
    linewidth=0.8
)

style_axes(ax)

for position, (_, row) in enumerate(plot_df.iterrows()):
    ax.text(
        row["Patent Count"] + 2,
        position,
        f'{int(row["Patent Count"])} '
        f'({row["Share (%)"]:.1f}%)',
        va="center",
        fontsize=8.5,
        color=grey
    )

fig.text(
    0.085,
    0.025,
    f"Source: Lens patent analysis; n={len(topic_df)} "
    "English-language patent families used in the topic model.",
    fontsize=7,
    color=grey
)

plt.subplots_adjust(
    left=0.34,
    right=0.96,
    top=0.88,
    bottom=0.16
)

plt.savefig(
    topic_distribution_figure,
    dpi=200,
    facecolor="white",
    bbox_inches="tight"
)

plt.close()


# topic 2 and 3 shares
recent_topics = topic_df[
    topic_df["Priority Year"].between(
        2021,
        2025
    )
].copy()

year_totals = (
    recent_topics
    .groupby("Priority Year")
    .size()
)

topic_year_counts = (
    recent_topics
    .groupby(
        [
            "Priority Year",
            "Top Topic"
        ]
    )
    .size()
    .unstack(fill_value=0)
)

topic_shares = (
    topic_year_counts
    .div(year_totals, axis=0)
    * 100
)

print("\nTopic 2 and 3 shares:")
print(
    topic_shares[
        [2, 3]
    ].round(1)
)

fig, ax = plt.subplots(
    figsize=(7.6, 5.2),
    dpi=200
)

ax.plot(
    topic_shares.index,
    topic_shares[2],
    color=blue,
    linewidth=2,
    marker="o",
    markersize=4.5,
    label="Membrane conductivity and stability"
)

ax.plot(
    topic_shares.index,
    topic_shares[3],
    color=orange,
    linewidth=2,
    marker="o",
    markersize=4.5,
    label="OER kinetics and catalyst stability"
)

ax.set_title(
    "Share of patent families in the two leading technical topics",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=title_color,
    pad=12
)

ax.set_ylabel("Share of patent families (%)")
ax.set_xlabel("")

ax.set_ylim(0, 35)
ax.set_yticks(range(0, 36, 5))
ax.set_xticks(topic_shares.index)

ax.grid(
    axis="y",
    color=grid_color,
    linewidth=0.8
)

style_axes(ax)

legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False,
    fontsize=8.5
)

for text in legend.get_texts():
    text.set_color(grey)

fig.text(
    0.085,
    0.025,
    "Source: Lens patent analysis; dominant-topic shares among "
    "English-language patents. Recent years may be affected by publication lag.",
    fontsize=7,
    color=grey
)

plt.subplots_adjust(
    left=0.12,
    right=0.985,
    top=0.88,
    bottom=0.25
)

plt.savefig(
    topic_share_figure,
    dpi=200,
    facecolor="white",
    bbox_inches="tight"
)

plt.close()


# technical review
topic_df["Technical Text"] = (
    topic_df["Title"].fillna("")
    + ". "
    + topic_df["Abstract"].fillna("")
)

topic_2_terms = {
    "chemical stability": [
        "chemical stability",
        "alkaline stability",
        "chemical degradation",
        "alkaline degradation",
        "degradation"
    ],
    "polymer backbone": [
        "backbone",
        "main chain",
        "polyaryl",
        "piperidine",
        "ether-free",
        "ether free"
    ],
    "crosslinking": [
        "crosslink",
        "cross-linked",
        "crosslinked"
    ],
    "swelling control": [
        "swelling",
        "dimensional stability",
        "water uptake"
    ],
    "mechanical stability": [
        "mechanical stability",
        "mechanical strength",
        "brittle",
        "brittleness"
    ],
    "ionic conductivity": [
        "ionic conductivity",
        "hydroxide conductivity",
        "anion conductivity",
        "conductivity",
        "ion transport"
    ],
    "crossover / permeability": [
        "crossover",
        "gas permeability",
        "hydrogen permeability",
        "oxygen permeability"
    ]
}

topic_3_terms = {
    "OER activity": [
        "oxygen evolution",
        "oer",
        "oxygen evolution reaction",
        "catalytic activity"
    ],
    "overpotential": [
        "overpotential"
    ],
    "high current density": [
        "current density",
        "ampere-level",
        "ampere level",
        "high current"
    ],
    "catalyst stability": [
        "catalyst stability",
        "stable catalyst",
        "long-term stability",
        "long term stability",
        "durability"
    ],
    "catalyst detachment": [
        "falling off",
        "detachment",
        "delamination",
        "adhesion"
    ],
    "nickel-based catalyst": [
        "nickel",
        "ni-fe",
        "nife",
        "nickel iron"
    ],
    "cobalt-based catalyst": [
        "cobalt",
        "co-fe",
        "cofe"
    ],
    "self-supported electrode": [
        "self-supported",
        "self supported",
        "foam",
        "porous substrate"
    ]
}

topic_2_review = (
    topic_df
    .sort_values(
        "Topic 2 Score",
        ascending=False
    )
    .head(30)
    .copy()
)

topic_3_review = (
    topic_df
    .sort_values(
        "Topic 3 Score",
        ascending=False
    )
    .head(30)
    .copy()
)

topic_2_review["Main Topic"] = (
    "Topic 2 - AEM polymer chemistry and membrane stability"
)

topic_3_review["Main Topic"] = (
    "Topic 3 - OER catalysts and catalytic electrodes"
)

topic_2_review["Technical Issues / Solutions"] = (
    topic_2_review["Technical Text"]
    .apply(
        lambda text: find_categories(
            text,
            topic_2_terms
        )
    )
)

topic_3_review["Technical Issues / Solutions"] = (
    topic_3_review["Technical Text"]
    .apply(
        lambda text: find_categories(
            text,
            topic_3_terms
        )
    )
)

technical_review = pd.concat(
    [
        topic_2_review,
        topic_3_review
    ],
    ignore_index=True
)

technical_review[
    [
        "Main Topic",
        "Lens ID",
        "Title",
        "Abstract",
        "Priority Year",
        "Topic 2 Score",
        "Topic 3 Score",
        "Technical Issues / Solutions"
    ]
].to_csv(
    technical_review_file,
    index=False
)

print("\nTopic 2 technical areas:")

for category in topic_2_terms:
    count = (
        topic_2_review[
            "Technical Issues / Solutions"
        ]
        .str.contains(
            category,
            regex=False
        )
        .sum()
    )

    print(category, ":", count)

print("\nTopic 3 technical areas:")

for category in topic_3_terms:
    count = (
        topic_3_review[
            "Technical Issues / Solutions"
        ]
        .str.contains(
            category,
            regex=False
        )
        .sum()
    )

    print(category, ":", count)


# european patent analysis
patent_df["Has EP Member"] = (
    patent_df[
        "Simple Family Member Jurisdictions"
    ]
    .fillna("")
    .astype(str)
    .str.contains(
        r"(?:^|[^A-Z])EP(?:[^A-Z]|$)",
        case=False,
        regex=True
    )
)

print(
    "\nEP-containing families:",
    patent_df["Has EP Member"].sum()
)

ep_topic_df = topic_df.merge(
    patent_df[
        [
            "Lens ID",
            "Has EP Member"
        ]
    ],
    on="Lens ID",
    how="left"
)

ep_topic_df["Has EP Member"] = (
    ep_topic_df["Has EP Member"]
    .fillna(False)
)

all_topic_counts = (
    ep_topic_df["Top Topic"]
    .value_counts()
    .reindex(
        range(1, 7),
        fill_value=0
    )
)

ep_counts = (
    ep_topic_df.loc[
        ep_topic_df["Has EP Member"],
        "Top Topic"
    ]
    .value_counts()
    .reindex(
        range(1, 7),
        fill_value=0
    )
)

ep_share = (
    ep_counts
    / all_topic_counts
    * 100
)

ep_results = pd.DataFrame({
    "Topic": range(1, 7),
    "Topic Name": [
        topic_names[topic]
        for topic in range(1, 7)
    ],
    "All Patents": all_topic_counts.values,
    "EP-containing Families": ep_counts.values,
    "EP Share (%)": ep_share.values
})

ep_results.to_csv(
    ep_topic_file,
    index=False
)

print("\nEP analysis:")
print(
    ep_results
    .round(1)
    .to_string(index=False)
)


# european patent figure
fig, ax = plt.subplots(
    figsize=(7.6, 5.2),
    dpi=200
)

ax.barh(
    ep_results["Topic Name"],
    ep_results["EP Share (%)"],
    color=blue,
    height=0.62
)

ax.invert_yaxis()

ax.set_title(
    "European patent protection by technical topic",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=title_color,
    pad=12
)

ax.set_xlabel("EP-containing families (%)")
ax.set_ylabel("")

ax.set_xlim(0, 30)
ax.set_xticks(range(0, 31, 5))

ax.grid(
    axis="x",
    color=grid_color,
    linewidth=0.8
)

style_axes(ax)

for index, row in ep_results.iterrows():
    ax.text(
        row["EP Share (%)"] + 0.5,
        index,
        f'{row["EP Share (%)"]:.1f}% '
        f'({int(row["EP-containing Families"])}/'
        f'{int(row["All Patents"])})',
        va="center",
        fontsize=8.5,
        color=grey
    )

fig.text(
    0.085,
    0.025,
    f"Source: Lens patent analysis; n={len(topic_df)} English-language "
    "patent families in the topic model. EP presence indicates that "
    "protection was sought via the EPO, not that a patent was granted.",
    fontsize=7,
    color=grey
)

plt.subplots_adjust(
    left=0.34,
    right=0.965,
    top=0.88,
    bottom=0.16
)

plt.savefig(
    ep_figure,
    dpi=200,
    facecolor="white",
    bbox_inches="tight"
)

plt.close()