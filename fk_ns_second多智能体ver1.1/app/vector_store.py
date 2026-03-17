import json
import math
import os
from typing import Any

import requests

from app.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    VECTOR_INDEX_LIMIT,
    VECTOR_STORE_PATH,
)
from app.database import search_news


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _clip_text(text: str, limit: int = 2000) -> str:
    return _normalize_text(text)[:limit]


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot = sum(left * right for left, right in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """公开的余弦相似度接口，便于其他模块复用向量匹配能力。"""
    return _cosine_similarity(vector_a, vector_b)


def _embedding_endpoint() -> str:
    return EMBEDDING_BASE_URL.rstrip("/") + "/embeddings"


def _embed_texts(texts: list[str]) -> list[list[float]]:
    normalized_texts = [_clip_text(text) for text in texts if _normalize_text(text)]
    if not normalized_texts:
        return []

    response = requests.post(
        _embedding_endpoint(),
        headers={
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": normalized_texts,
            "encoding_format": "float",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    vectors = [item.get("embedding", []) for item in sorted(data, key=lambda item: item.get("index", 0))]
    if len(vectors) != len(normalized_texts):
        raise RuntimeError("向量接口返回数量与输入数量不一致")
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """公开的文本向量化接口。"""
    return _embed_texts(texts)


def batch_semantic_similarity(query_text: str, candidate_texts: list[str]) -> list[float]:
    """批量计算查询与候选文本之间的语义相似度。"""
    normalized_candidates = [_clip_text(text) for text in candidate_texts]
    if not _normalize_text(query_text) or not normalized_candidates:
        return [0.0 for _ in candidate_texts]

    vectors = _embed_texts([query_text, *normalized_candidates])
    if len(vectors) != len(normalized_candidates) + 1:
        raise RuntimeError("批量语义相似度计算返回向量数量异常")

    query_vector = vectors[0]
    return [max(0.0, _cosine_similarity(query_vector, vector)) for vector in vectors[1:]]


def _load_store() -> dict:
    if not os.path.exists(VECTOR_STORE_PATH):
        return {"items": [], "metadata": {}}
    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_store(payload: dict) -> None:
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def rebuild_vector_store(limit: int = VECTOR_INDEX_LIMIT) -> dict:
    rows = search_news(limit=limit)
    items = []
    texts = []
    for row in rows:
        text = _clip_text(row.get("news_text", ""))
        if not text:
            continue
        items.append(
            {
                "id": row.get("id"),
                "news_text": row.get("news_text", ""),
                "is_real": row.get("is_real"),
                "reason": row.get("reason", ""),
                "evidence_url": row.get("evidence_url", ""),
            }
        )
        texts.append(text)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        vectors.extend(_embed_texts(batch))

    stored_items = []
    for item, vector in zip(items, vectors):
        stored_items.append({**item, "vector": vector})

    payload = {
        "metadata": {
            "model": EMBEDDING_MODEL,
            "limit": limit,
            "count": len(stored_items),
            "latest_ids": [item["id"] for item in stored_items],
        },
        "items": stored_items,
    }
    _save_store(payload)
    return payload


def ensure_vector_store(limit: int = VECTOR_INDEX_LIMIT) -> dict:
    payload = _load_store()
    rows = search_news(limit=limit)
    latest_ids = [row.get("id") for row in rows if row.get("id") is not None]
    metadata = payload.get("metadata", {})
    if (
        not payload.get("items")
        or metadata.get("model") != EMBEDDING_MODEL
        or metadata.get("limit") != limit
        or metadata.get("latest_ids") != latest_ids
    ):
        return rebuild_vector_store(limit=limit)
    return payload


def search_similar_news(news_text: str, top_k: int = 20, limit: int = VECTOR_INDEX_LIMIT) -> list[dict[str, Any]]:
    payload = ensure_vector_store(limit=limit)
    items = payload.get("items", [])
    if not items:
        return []

    query_vectors = _embed_texts([news_text])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    scored_items = []
    for item in items:
        score = _cosine_similarity(query_vector, item.get("vector", []))
        if score < 0.35:
            continue
        scored_items.append(
            {
                "id": item.get("id"),
                "score": round(float(score), 4),
                "news_text": item.get("news_text", ""),
                "is_real": item.get("is_real"),
                "reason": item.get("reason", ""),
                "evidence_url": item.get("evidence_url", ""),
                "match_type": "vector",
            }
        )

    scored_items.sort(key=lambda item: item["score"], reverse=True)
    return scored_items[:top_k]


def upsert_vector_entry(news_id: int, news_text: str, is_real: Any, reason: str, evidence_url: str, limit: int = VECTOR_INDEX_LIMIT) -> dict:
    payload = _load_store()
    items = [item for item in payload.get("items", []) if item.get("id") != news_id]
    vectors = _embed_texts([news_text])
    if not vectors:
        raise RuntimeError("无法为新增新闻生成向量")

    items.insert(
        0,
        {
            "id": news_id,
            "news_text": news_text,
            "is_real": is_real,
            "reason": reason,
            "evidence_url": evidence_url,
            "vector": vectors[0],
        },
    )
    items = items[:limit]

    payload = {
        "metadata": {
            "model": EMBEDDING_MODEL,
            "limit": limit,
            "count": len(items),
            "latest_ids": [item.get("id") for item in items],
        },
        "items": items,
    }
    _save_store(payload)
    return payload