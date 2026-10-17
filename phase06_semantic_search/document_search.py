from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearchEngine:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_embeddings = None

    def index_documents(self, documents: list[str]):
        self.documents = documents
        self.document_embeddings = self.model.encode(documents)
        print(f"Indexed {len(documents)} documents. Embedding shape: {self.document_embeddings.shape}")

    def search(self, query: str, top_k: int = None) -> list[tuple[str, float]]:
        if not self.documents:
            raise ValueError("No documents indexed. Call index_documents() first.")

        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.document_embeddings)[0]

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        if top_k is not None:
            return ranked[:top_k]
        return ranked


def run():
    print("=" * 60)
    print("PHASE 06: SEMANTIC SEARCH ENGINE")
    print("=" * 60)

    documents = [
        "Students can register courses through the student portal.",
        "The examination office publishes examination schedules.",
        "The finance office handles tuition fee issues.",
        "The library provides digital research resources.",
    ]

    engine = SemanticSearchEngine()
    engine.index_documents(documents)

    test_queries = [
        "Where can I register my courses?",
        "When will final exam timetable and date sheets be announced?",
        "I need help resolving my university tuition fee balance.",
        "How do I access academic journals, books, and study materials?",
    ]

    print("\n" + "=" * 60)
    print("TESTING SEMANTIC SEARCH WITH MULTIPLE USER QUERIES")
    print("=" * 60)

    for idx, query in enumerate(test_queries, 1):
        print(f"\n[QUERY {idx}]: \"{query}\"")
        results = engine.search(query)
        for rank, (doc, score) in enumerate(results, 1):
            star = "★ (BEST MATCH)" if rank == 1 else " "
            print(f"  Rank {rank} | Score: {score:.4f} {star} -> {doc}")


if __name__ == "__main__":
    run()
