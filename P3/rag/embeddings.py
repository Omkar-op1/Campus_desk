from __future__ import annotations

from collections.abc import Callable
from typing import Any


class GeminiEmbeddingFunction:
    def __init__(
        self,
        client: Any,
        *,
        model: str = "gemini-embedding-2",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self.client = client
        self.model = model
        self.task_type = task_type

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config={"task_type": self.task_type},
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError("Gemini returned an incomplete embedding batch.")
        values: list[list[float]] = []
        for embedding in embeddings:
            vector = getattr(embedding, "values", None)
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("Gemini returned an empty embedding.")
            values.append([float(value) for value in vector])
        return values


def create_embedding_functions(
    client: Any,
    *,
    model: str = "gemini-embedding-2",
) -> tuple[Callable[[list[str]], list[list[float]]], Callable[[list[str]], list[list[float]]]]:
    return (
        GeminiEmbeddingFunction(
            client,
            model=model,
            task_type="RETRIEVAL_DOCUMENT",
        ),
        GeminiEmbeddingFunction(
            client,
            model=model,
            task_type="RETRIEVAL_QUERY",
        ),
    )
