import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF


# files
project_root = Path(__file__).resolve().parent

input_file = project_root / "data" / "aem_patents_cleaned.csv"
output_dir = project_root / "outputs" / "tables"

output_dir.mkdir(parents=True, exist_ok=True)

# settings
topic_numbers = [5, 6, 7]
n_top_words = 15
n_representative_patents = 5

# load data
df = pd.read_csv(input_file)

print("Rows loaded:", len(df))

if "Text Clean" not in df.columns:
    raise ValueError("Missing required column: Text Clean")

# stopwords
patent_stopwords = {
    "method",
    "methods",
    "preparation",
    "prepared",
    "prepare",
    "application",
    "applications",
    "invention",
    "present",
    "comprising",
    "comprise",
    "comprises",
    "thereof",
    "manufacturing",
    "manufacture",
    "manufactured",
    "based",
    "use",
    "uses",
    "using",
    "used",
    "provided",
    "provides",
    "providing"
}

custom_stopwords = list(
    set(ENGLISH_STOP_WORDS).union(patent_stopwords)
)

# text
texts = df["Text Clean"].fillna("")

# tf-idf
vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85
)

tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# run topic models
for n_topics in topic_numbers:

    print(f"\nMODEL WITH {n_topics} TOPICS")

    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=500
    )

    topic_matrix = nmf_model.fit_transform(tfidf_matrix)

    # top words
    print("\nTop words:")

    for topic_number, topic in enumerate(
        nmf_model.components_,
        start=1
    ):
        top_indices = topic.argsort()[-n_top_words:][::-1]

        top_words = [
            feature_names[index]
            for index in top_indices
        ]

        print(
            f"\nTopic {topic_number}: "
            + ", ".join(top_words)
        )

    result_df = df.copy()

    # dominant topic
    result_df["Dominant Topic"] = (
        topic_matrix.argmax(axis=1) + 1
    )

    result_df["Topic Strength"] = (
        topic_matrix.max(axis=1)
    )

    print("\nPatent count per topic:")
    print(
        result_df["Dominant Topic"]
        .value_counts()
        .sort_index()
    )

    # representative patents
    print("\nRepresentative patents:")

    for topic_number in range(1, n_topics + 1):

        topic_scores = topic_matrix[:, topic_number - 1]

        top_indices = (
            topic_scores
            .argsort()[-n_representative_patents:][::-1]
        )

        print(f"\nTopic {topic_number}:")

        for index in top_indices:
            title = result_df.iloc[index]["Title"]
            score = topic_scores[index]

            print(f"- {title} | score: {score:.4f}")

    # save topic scores
    for topic_number in range(1, n_topics + 1):
        result_df[f"Topic {topic_number} Score"] = (
            topic_matrix[:, topic_number - 1]
        )

    # save
    output_file = output_dir / f"aem_topics_{n_topics}.csv"

    result_df.to_csv(
        output_file,
        index=False
    )

    print("Saved:", output_file)