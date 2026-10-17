import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_dataset(path: str):
    df = pd.read_csv(path)
    return df["message"], df["label"]


def train_classifier(path: str):
    texts, labels = load_dataset(path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    print("Pipeline Classification Report (Support Tickets):")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    test_queries = [
        "I forgot my password and portal login failed",
        "The backend server is throwing error 500",
        "Where should I pay my semester fee receipt?",
        "When is registration deadline for new admissions?",
    ]

    print("\nInference with Pipeline:")
    for query in test_queries:
        predicted_label = pipeline.predict([query])[0]
        print(f"  Query: '{query}' -> [{predicted_label}]")

    return pipeline


def run():
    print("=" * 60)
    print("PHASE 04 - CODE 01: SCIKIT-LEARN PIPELINE CLASSIFICATION")
    print("=" * 60)

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "support_tickets.csv"),
        os.path.join("data", "support_tickets.csv"),
        "support_tickets.csv",
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        raise FileNotFoundError("Could not find support_tickets.csv in candidate locations.")

    print(f"Dataset Loaded From: {csv_path}")
    train_classifier(csv_path)


if __name__ == "__main__":
    run()
