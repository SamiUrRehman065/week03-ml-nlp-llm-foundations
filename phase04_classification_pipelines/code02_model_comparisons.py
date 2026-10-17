from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def run():
    print("=" * 60)
    print("PHASE 04 - CODE 02: TF-IDF & MODEL COMPARISONS")
    print("=" * 60)

    texts = [
        "I love this product",
        "This product is excellent",
        "Amazing experience",
        "I really enjoyed this",
        "This is fantastic",
        "I hate this product",
        "This product is terrible",
        "Very bad experience",
        "I did not enjoy this",
        "This is horrible",
    ]

    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(),
        "LinearSVC": LinearSVC(),
        "MultinomialNB": MultinomialNB(),
    }

    new_text = ["I really love this"]
    new_vectors = vectorizer.transform(new_text)

    print(f"Dataset Size: {len(texts)} samples (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Features (TF-IDF Vocab): {len(vectorizer.get_feature_names_out())} tokens\n")

    for name, model in models.items():
        print("-" * 50)
        print(f"MODEL: {name}")
        print("-" * 50)

        model.fit(X_train_vectors, y_train)
        predictions = model.predict(X_test_vectors)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        pred_new = model.predict(new_vectors)
        print(f"Inference on '{new_text[0]}': {pred_new[0]}")


if __name__ == "__main__":
    run()
