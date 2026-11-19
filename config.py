
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_google_api_key():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "Please configure your .env file."
        )

    return api_key


def get_app_config():

    return {
        "app_name": "DocuMind AI",
        "embedding_model": "gemini-embedding-001",
        "llm_model": "gemini-2.5-flash",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4
    }
