from __future__ import annotations

import re
from typing import Iterable, Optional

from sklearn.feature_extraction.text import TfidfVectorizer


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def is_free_text(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if contains_cjk(normalized):
        return len(normalized) >= 6
    return len(normalized.split()) >= 3 or len(normalized) >= 24


def use_character_ngrams(texts: Iterable[str]) -> bool:
    normalized = [str(text or "") for text in texts]
    if any(contains_cjk(text) for text in normalized):
        return True

    token_counts = [len(text.split()) for text in normalized if text.strip()]
    if not token_counts:
        return False
    return (sum(token_counts) / len(token_counts)) < 2.5


def build_tfidf_vectorizer(
    texts: Iterable[str],
    *,
    max_features: Optional[int] = None,
    stop_words: Optional[str] = "english",
) -> TfidfVectorizer:
    if use_character_ngrams(texts):
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            lowercase=True,
            max_features=max_features,
        )

    kwargs = {
        "lowercase": True,
        "ngram_range": (1, 2),
    }
    if stop_words is not None:
        kwargs["stop_words"] = stop_words
    if max_features is not None:
        kwargs["max_features"] = max_features
    return TfidfVectorizer(**kwargs)
