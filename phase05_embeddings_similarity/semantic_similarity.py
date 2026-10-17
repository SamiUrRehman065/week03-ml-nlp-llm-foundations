from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def run():
    print("=" * 60)
    print("PHASE 05: EMBEDDINGS & SEMANTIC SIMILARITY")
    print("=" * 60)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading Model: {model_name}...")
    model = SentenceTransformer(model_name)

    sentences = [
        "How do I reset my password?",
        "I forgot my password.",
        "The weather is very hot today.",
    ]

    print("\nInput Sentences:")
    for idx, s in enumerate(sentences):
        print(f"  [{idx}] '{s}'")

    embeddings = model.encode(sentences)
    print(f"\nGenerated Embeddings Shape: {embeddings.shape} (Count: {embeddings.shape[0]}, Dimension: {embeddings.shape[1]})")

    password_similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    weather_similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[2]],
    )[0][0]

    print(f"\nPassword similarity [Sentence 0 vs Sentence 1]: {password_similarity:.4f}")
    print(f"Weather similarity  [Sentence 0 vs Sentence 2]: {weather_similarity:.4f}")

    additional_sentences = [
        ("The laptop battery is dying quickly.", "My computer discharges in less than an hour."),
        ("Students study for exam.", "Children are playing football in the garden."),
        ("Tuition fees must be paid by Friday.", "When is the tuition fee payment deadline?"),
    ]

    print("\n--- EXTENDED PAIRWISE COMPARISONS ---")
    for s1, s2 in additional_sentences:
        emb = model.encode([s1, s2])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        print(f"  Pair: \"{s1}\"\n        \"{s2}\"\n  -> Cosine Similarity: {sim:.4f}\n")

    print("--- THEORETICAL SUMMARY: WORD COUNTS VS. SEMANTIC EMBEDDINGS ---")
    print("1. Word Counts / TF-IDF (Sparse Representations):")
    print("   - Treat words as orthogonal dimensions in a massive, sparse vocabulary space.")
    print("   - Lack semantic awareness: 'forgot password' and 'reset password' have 0 overlap")
    print("     on those specific differing words, yielding low lexical similarity despite near-identical intent.")
    print("   - Suffer from the curse of dimensionality and cannot capture synonyms or context.")
    print("\n2. Dense Semantic Embeddings (Sentence Transformers):")
    print("   - Encode text into continuous, low-dimensional vector space (e.g. 384 or 768 dimensions).")
    print("   - Capture deep semantic meaning, context, and intent learned via contrastive pretraining.")
    print("   - Sentences sharing meaning map to vectors pointing in almost the same direction")
    print("     (high cosine similarity near 1.0), even with completely different vocabularies.")


if __name__ == "__main__":
    run()
