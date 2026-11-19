
import os
from vector_store import load_vector_store


def retrieve_documents(query, save_path, top_k=4):

    if not query or not query.strip():
        raise ValueError("Please enter a valid question.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    index_path = os.path.join(save_path, "index.faiss")
    metadata_path = os.path.join(save_path, "index.pkl")

    if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError("FAISS database not found.")

    vector_db = load_vector_store(save_path)

    results = vector_db.similarity_search_with_score(
        query.strip(),
        k=top_k
    )

    return [
        {
            "text": document.page_content,
            "page_number": document.metadata.get("page_number", "Unknown"),
            "score": float(score)
        }
        for document, score in results
    ]
