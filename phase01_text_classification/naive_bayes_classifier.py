from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def run_original_phase01():
    print("=" * 60)
    print("PHASE 01: BASELINE NAIVE BAYES CLASSIFICATION (ORIGINAL CODE)")
    print("=" * 60)

    texts = [
        "I love this course",
        "This course is excellent",
        "I hate this course",
        "This course is terrible",
    ]

    labels = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    new_text = ["I really love this course"]
    new_X = vectorizer.transform(new_text)
    prediction = model.predict(new_X)

    print("Training Vocabulary:", vectorizer.get_feature_names_out().tolist())
    print("Input Text:", new_text[0])
    print("Prediction:", prediction[0])
    return model, vectorizer


def run_comprehensive_evaluation():
    print("\n" + "=" * 60)
    print("PHASE 01: COMPREHENSIVE DATASET, TRAIN/TEST SPLIT & EVALUATION")
    print("=" * 60)

    texts = [
        "I love this course",
        "This course is excellent",
        "Fantastic teacher and wonderful material",
        "Great content and very well explained",
        "Highly recommended educational experience",
        "I hate this course",
        "This course is terrible",
        "Horrible presentation and complete waste of time",
        "Very poor explanations and confusing slides",
        "Awful quality and very disappointing lecture",
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

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"Dataset Size: {len(texts)} samples (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    test_queries = [
        "I really love this course",
        "The course was terrible and useless",
        "Fantastic explanation of concepts",
    ]
    print("\nInference on New Unseen Texts:")
    query_vecs = vectorizer.transform(test_queries)
    preds = model.predict(query_vecs)
    probs = model.predict_proba(query_vecs)

    for query, pred, prob in zip(test_queries, preds, probs):
        pos_idx = list(model.classes_).index("positive")
        neg_idx = list(model.classes_).index("negative")
        print(f"  - '{query}' -> [{pred.upper()}] (pos_prob={prob[pos_idx]:.2f}, neg_prob={prob[neg_idx]:.2f})")

    print("\n--- THEORETICAL SUMMARY: TRAINING VS. INFERENCE ---")
    print("1. Training: The phase where the model learns statistical parameters/weights from labeled data.")
    print("   Here, CountVectorizer learns the vocabulary dictionary and document frequencies (fit_transform),")
    print("   and MultinomialNB calculates prior and likelihood probabilities P(w|c).")
    print("2. Inference: The phase where the trained model processes new, unseen text using fixed parameters.")
    print("   CountVectorizer converts the new text using the previously fitted vocabulary (transform),")
    print("   and the classifier evaluates argmax_c P(c|text) to output the predicted class without updating weights.")


if __name__ == "__main__":
    run_original_phase01()
    run_comprehensive_evaluation()
