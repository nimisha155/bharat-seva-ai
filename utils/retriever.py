"""
Semantic retrieval layer for Bharat Seva AI.

This module loads the government scheme knowledge base from
data/schemes.json, converts each scheme into embeddings using Google's
Gemini embedding model, and lets the rest of the app retrieve the most
relevant schemes for a natural-language user query using cosine
similarity.

This module does not talk to Streamlit's chat UI directly. It exposes
plain functions that app.py (or any other caller) can use.
"""

import json
import os

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# ---------- Configuration ----------
SCHEMES_FILE_PATH = os.path.join("data", "schemes.json")
EMBEDDING_MODEL = "gemini-embedding-001"

# Minimum cosine similarity for a scheme to be considered relevant.
# Schemes scoring below this are treated as unrelated to the query.
SIMILARITY_THRESHOLD = 0.5


# ---------- Custom exceptions ----------
class RetrieverError(Exception):
    """Base exception for retrieval-layer failures. Kept generic and
    descriptive so calling code (e.g. app.py) can catch it and show a
    friendly message without needing to know internal details."""
    pass


# ---------- Gemini client ----------
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """
    Creates a Gemini client using the API key from Streamlit Secrets.
    Cached as a resource so the same client is reused across reruns.
    Raises RetrieverError if the key is missing.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RetrieverError(
            "GEMINI_API_KEY is missing from Streamlit Secrets. "
            "Add it to your app's secrets to enable scheme retrieval."
        )
    return genai.Client(api_key=api_key)


# ---------- Loading the knowledge base ----------
@st.cache_data(show_spinner=False)
def load_schemes(path: str = SCHEMES_FILE_PATH):
    """
    Loads and parses data/schemes.json.
    Returns the list of scheme dictionaries.
    Raises RetrieverError on missing file or invalid JSON.
    """
    if not os.path.exists(path):
        raise RetrieverError(f"Scheme data file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RetrieverError(f"Scheme data file is not valid JSON: {e}")

    schemes = data.get("schemes")
    if schemes is None:
        raise RetrieverError(
            "Scheme data file is missing the top-level 'schemes' array."
        )

    return schemes


def create_scheme_text(scheme: dict) -> str:
    """
    Builds a single searchable text block for a scheme, combining the
    fields most useful for matching a user's natural-language question.
    This text is what gets embedded, not the raw JSON.
    """
    parts = [
        f"Scheme name: {scheme.get('name', '')}",
        f"Category: {scheme.get('category', '')}",
        f"Description: {scheme.get('description', '')}",
        f"Benefits: {scheme.get('benefits', '')}",
        f"Eligibility: {scheme.get('eligibility', '')}",
        f"Target beneficiaries: {scheme.get('target_beneficiaries', '')}",
    ]
    return "\n".join(parts)


# ---------- Embedding helpers ----------
def _embed_texts(client, texts, task_type: str):
    """
    Calls the Gemini embedding API for a list of texts and returns a
    list of embedding vectors (as Python lists of floats).

    task_type should be "RETRIEVAL_DOCUMENT" for scheme texts or
    "RETRIEVAL_QUERY" for the user's question, per Gemini embedding
    model guidance for asymmetric retrieval.
    """
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type)
        )
    except Exception as e:
        raise RetrieverError(f"Embedding API call failed: {e}")

    if not response or not getattr(response, "embeddings", None):
        raise RetrieverError("Embedding API returned no embeddings.")

    return [list(e.values) for e in response.embeddings]


@st.cache_resource(show_spinner="Preparing scheme knowledge base...")
def get_scheme_embeddings():
    """
    Loads all schemes, builds their searchable text, and generates one
    embedding per scheme. Cached as a resource for the lifetime of the
    app process so this expensive step runs only once, not on every
    user question.

    Returns a tuple of (schemes, embeddings_matrix) where
    embeddings_matrix is a NumPy array of shape (num_schemes, dim).
    """
    schemes = load_schemes()
    client = get_gemini_client()

    scheme_texts = [create_scheme_text(s) for s in schemes]
    embeddings = _embed_texts(client, scheme_texts, task_type="RETRIEVAL_DOCUMENT")

    return schemes, np.array(embeddings, dtype=np.float32)


def embed_query(query: str):
    """
    Generates an embedding for a single user query.
    Raises RetrieverError on empty query or API failure.
    """
    if not query or not query.strip():
        raise RetrieverError("Cannot embed an empty query.")

    client = get_gemini_client()
    embeddings = _embed_texts(client, [query.strip()], task_type="RETRIEVAL_QUERY")
    return np.array(embeddings[0], dtype=np.float32)


# ---------- Similarity ----------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Returns the cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------- Main retrieval function ----------
def retrieve_schemes(query: str, top_k: int = 3, threshold: float = SIMILARITY_THRESHOLD):
    """
    Returns the most relevant schemes for a natural-language query.

    Args:
        query: The user's natural-language question.
        top_k: Maximum number of schemes to return.
        threshold: Minimum cosine similarity for a scheme to be
            included. Schemes scoring below this are dropped, so
            irrelevant queries can return an empty list rather than
            forcing a match.

    Returns:
        A list of dicts, each shaped like:
        {"scheme": {...}, "similarity": 0.82}
        sorted by similarity in descending order.

    Raises:
        RetrieverError on empty query, missing/invalid data file,
        missing API key, or embedding API failure.
    """
    if not query or not query.strip():
        raise RetrieverError("Query must not be empty.")

    schemes, scheme_embeddings = get_scheme_embeddings()
    query_embedding = embed_query(query)

    similarities = [
        cosine_similarity(query_embedding, scheme_embeddings[i])
        for i in range(len(schemes))
    ]

    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        score = similarities[idx]
        if score < threshold:
            continue
        results.append({"scheme": schemes[idx], "similarity": round(score, 4)})

    return results
