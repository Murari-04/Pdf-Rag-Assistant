import re
import time
import random

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import get_google_api_key, get_app_config

# Get Gemini embedding model
def get_embedding_model():

    config = get_app_config()

    return GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"],
        google_api_key=get_google_api_key(),
        max_retries=2
    )

# Check whether an error can be retried
def is_retryable_error(error):

    message = str(error).upper()

    return any(code in message for code in [
        "429", "RESOURCE_EXHAUSTED", "RATE LIMIT",
        "500", "502", "503", "504",
        "TIMEOUT", "SERVICE UNAVAILABLE"
    ])


# Calculate retry delay
def get_retry_delay(error, attempt):

    message = str(error)

    match = re.search(
        r"(?:retryDelay['\"]?\s*:\s*['\"]?|retry in )(\d+(?:\.\d+)?)s",
        message,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1)) + 1

    return min(15 * (2 ** attempt), 120) + random.uniform(0, 3)

# Generate embeddings with retry handling
def embed_documents_with_retry(
    text_chunks,
    embedding_model=None,
    max_retries=4
):

    if not text_chunks:
        raise ValueError("No text chunks provided.")

    if embedding_model is None:
        embedding_model = get_embedding_model()

    for attempt in range(max_retries):

        try:
            embeddings = embedding_model.embed_documents(text_chunks)

            if len(embeddings) != len(text_chunks):
                raise ValueError("Embedding count mismatch.")

            return embeddings

        except Exception as error:

            if not is_retryable_error(error):
                raise RuntimeError(
                    f"Embedding failed: {error}"
                ) from error

            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Embedding failed after {max_retries} attempts: {error}"
                ) from error

            wait = get_retry_delay(error, attempt)

            print(
                f"Gemini temporarily unavailable. "
                f"Retry {attempt + 1}/{max_retries}. "
                f"Waiting {wait:.1f} seconds..."
            )

            time.sleep(wait)

# Simple embedding function
def generate_embeddings(text_chunks):

    return embed_documents_with_retry(text_chunks)
