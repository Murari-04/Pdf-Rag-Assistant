import os
import time
from collections import deque

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embedding_generator import (
    get_embedding_model,
    embed_documents_with_retry
)

# Embedding configuration
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
BATCH_DELAY = float(os.getenv("EMBEDDING_BATCH_DELAY", "5"))
TOKEN_BUDGET = int(
    os.getenv("EMBEDDING_TOKEN_BUDGET_PER_MINUTE", "24000")
)

WINDOW = 60

# Estimate tokens
def estimate_tokens(text):
    return max(1, (len(text) + 2) // 3)

# Control API request rate
def wait_for_capacity(batch_tokens, recent_batches, last_batch_time):

    while True:

        now = time.monotonic()

        while recent_batches and now - recent_batches[0][0] >= WINDOW:
            recent_batches.popleft()

        used_tokens = sum(tokens for _, tokens in recent_batches)

        delay = (
            BATCH_DELAY - (now - last_batch_time)
            if last_batch_time is not None else 0
        )

        if delay > 0:
            time.sleep(delay)
            continue

        if used_tokens + batch_tokens > TOKEN_BUDGET and recent_batches:

            wait = WINDOW - (now - recent_batches[0][0]) + 1

            print(f"Token budget reached. Waiting {wait:.1f} seconds...")
            time.sleep(max(1, wait))
            continue

        return

# Create FAISS database
def create_vector_store(chunks, save_path="vectorstore"):

    if not chunks:
        raise ValueError("No chunks available for vector storage.")

    if BATCH_SIZE < 1 or BATCH_DELAY < 0 or TOKEN_BUDGET < 1:
        raise ValueError("Invalid embedding configuration.")

    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk.get("metadata", {})
        )
        for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not documents:
        raise ValueError("No valid document content found.")

    embedding_model = get_embedding_model()

    total = len(documents)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    all_embeddings = []
    recent_batches = deque()
    last_batch_time = None

    print(f"\nStarting embedding: {total} chunks")
    print(f"Batch size: {BATCH_SIZE} | Total batches: {batches}")

    for batch_number, start in enumerate(
        range(0, total, BATCH_SIZE), 1
    ):

        batch = documents[start:start + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]

        batch_tokens = sum(estimate_tokens(text) for text in texts)

        wait_for_capacity(
            batch_tokens,
            recent_batches,
            last_batch_time
        )

        print(f"Processing batch {batch_number}/{batches}...")

        embeddings = embed_documents_with_retry(
            texts,
            embedding_model=embedding_model,
            max_retries=4
        )

        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding count mismatch.")

        all_embeddings.extend(embeddings)

        last_batch_time = time.monotonic()
        recent_batches.append((last_batch_time, batch_tokens))

        print(f"Embedded {len(all_embeddings)}/{total} chunks")

    if len(all_embeddings) != total:
        raise RuntimeError("Embedding generation incomplete.")

    # Build FAISS database
    vector_db = FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, embedding)
            for doc, embedding in zip(documents, all_embeddings)
        ],
        embedding=embedding_model,
        metadatas=[doc.metadata for doc in documents]
    )

    os.makedirs(save_path, exist_ok=True)
    vector_db.save_local(save_path)

    print("\nFAISS database created successfully!")
    print(f"Vectors: {vector_db.index.ntotal}")
    print(f"Dimensions: {vector_db.index.d}")
    print(f"Location: {save_path}")

    return vector_db

# Load existing FAISS database
def load_vector_store(save_path="vectorstore"):

    if not all(
        os.path.exists(os.path.join(save_path, filename))
        for filename in ("index.faiss", "index.pkl")
    ):
        raise FileNotFoundError("FAISS database not found.")

    vector_db = FAISS.load_local(
        save_path,
        get_embedding_model(),
        allow_dangerous_deserialization=True
    )

    print(f"FAISS database loaded: {save_path}")

    return vector_db
