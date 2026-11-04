from __future__ import annotations

from collections.abc import Callable

import httpx


class VoyageEmbeddingFunction:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "voyage-4-lite",
        input_type: str = "document",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.input_type = input_type

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "input": texts,
                "model": self.model,
                "input_type": self.input_type,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("data")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Voyage returned an incomplete embedding batch.")
        embeddings.sort(key=lambda item: item.get("index", -1))
        values = [item.get("embedding") for item in embeddings]
        if any(not isinstance(vector, list) or not vector for vector in values):
            raise RuntimeError("Voyage returned an empty embedding.")
        return [[float(value) for value in vector] for vector in values]


def create_embedding_functions(
    api_key: str,
    *,
    model: str = "voyage-4-lite",
) -> tuple[Callable[[list[str]], list[list[float]]], Callable[[list[str]], list[list[float]]]]:
    return (
        VoyageEmbeddingFunction(
            api_key,
            model=model,
            input_type="document",
        ),
        VoyageEmbeddingFunction(
            api_key,
            model=model,
            input_type="query",
        ),
    )
