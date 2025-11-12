"""Skeleton helpers for normalizing ingredient text with Ollama."""

from __future__ import annotations

import unicodedata
from typing import Iterable, List

import requests

SYSTEM = """You normalize cooking ingredient mentions.
- Remove quantities, units, sizes, containers, prep words, and quality adjectives.
- Keep the ingredient, singular, generic.
- Merge US/UK synonyms (cilantro→coriander, arugula→rocket).
- Keep multi-word ingredients (e.g., brown sugar).
- If not an ingredient, output NONE.
- Output ONLY the normalized ingredient string, no punctuation or quotes.
"""

FEWSHOTS: list[tuple[str, str]] = [
    ("Pinch of crushed red pepper flakes", "red pepper flakes"),
    ("Freshly ground black pepper", "black pepper"),
    ('⅓ loaf good-quality sturdy white bread, torn into 1" pieces (about 2½ cups)', "white bread"),
]

DEFAULT_MODEL = "phi3:mini"
DEFAULT_API_URL = "http://localhost:11434/api/chat"


def preclean(text: str) -> str:
    """Trim, lowercase, and strip diacritics before invoking the model."""
    trimmed = text.strip().lower()
    normalized = unicodedata.normalize("NFKD", trimmed)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_with_llm(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    num_predict: int = 16,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 120.0,
) -> str:
    """Route `text` through Ollama using the baked-in system/few-shot prompt."""
    payload = {
        "model": model,
        "messages": _build_messages(text),
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return _extract_last_message(data).strip()


def _build_messages(user_text: str) -> list[dict[str, str]]:
    blocks: List[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    for user_value, assistant_value in FEWSHOTS:
        blocks.append({"role": "user", "content": user_value})
        blocks.append({"role": "assistant", "content": assistant_value})
    blocks.append({"role": "user", "content": user_text})
    return blocks


def _extract_last_message(payload: dict) -> str:
    """Return the assistant message content from the REST response body."""
    message = payload.get("message") or {}
    content = message.get("content")
    return content or ""


def main(example_texts: Iterable[str] | None = None) -> None:
    """Simple driver for ad-hoc testing."""
    samples = list(example_texts or (text for text, _ in FEWSHOTS))
    if not samples:
        print("No examples provided.")
        return
    for sample in samples:
        cleaned = preclean(sample)
        result = normalize_with_llm(cleaned)
        print(f"{sample} -> {result}")


if __name__ == "__main__":
    main()
