from __future__ import annotations

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_embedding_function = DefaultEmbeddingFunction()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embedding.tolist() for embedding in _embedding_function(texts)]
