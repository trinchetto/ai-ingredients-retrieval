from __future__ import annotations


import pytest
import requests

from ai_ingredients import ollama_normalizer as on


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("  Fresh Basil ", "fresh basil"),
        ("CILANTRO", "cilantro"),
        ("Crème brûlée", "creme brulee"),
    ],
)
def test_preclean_normalizes_whitespace_case_and_diacritics(
    raw_text: str, expected: str
) -> None:
    """Ensure preclean trims, lowercases, and strips accents consistently."""
    assert on.preclean(raw_text) == expected


@pytest.mark.parametrize("user_text", ["tomato sauce", "handful of almonds"])
def test_build_messages_appends_user_prompt_after_fewshots(user_text: str) -> None:
    """Validate _build_messages adds system, few-shot, then the user prompt."""
    messages = on._build_messages(user_text)

    assert messages[0] == {"role": "system", "content": on.SYSTEM}
    assert messages[-1] == {"role": "user", "content": user_text}
    assert any(msg["role"] == "assistant" for msg in messages[1:-1])


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"message": {"content": "basil"}}, "basil"),
        ({"message": {"content": ""}}, ""),
        ({}, ""),
        ({"message": {}}, ""),
    ],
)
def test_extract_last_message_handles_missing_content(
    payload: dict, expected: str
) -> None:
    """Confirm _extract_last_message is resilient to absent content."""
    assert on._extract_last_message(payload) == expected


class _DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.raise_was_called = False

    def raise_for_status(self) -> None:
        self.raise_was_called = True
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


@pytest.mark.parametrize(
    ("text", "response_content"),
    [
        ("fresh basil", "basil"),
        (" diced tomatoes ", "tomato"),
    ],
)
def test_normalize_with_llm_posts_payload_and_returns_trimmed_result(
    text: str,
    response_content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure normalize_with_llm sends the correct payload and parses the content."""
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict, timeout: float):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse({"message": {"content": f" {response_content} "}})

    monkeypatch.setattr(on.requests, "post", fake_post)

    result = on.normalize_with_llm(
        text,
        model="model-a",
        temperature=0.3,
        num_predict=8,
        api_url="http://api",
        timeout=10.0,
    )

    assert result == response_content
    assert captured["url"] == "http://api"
    assert captured["timeout"] == 10.0
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "model-a"
    assert payload["options"] == {"temperature": 0.3, "num_predict": 8}
    assert payload["messages"][-1]["content"] == text


def test_normalize_with_llm_raises_when_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure normalize_with_llm propagates HTTP errors from the client."""

    def fake_post(*_: object, **__: object):
        return _DummyResponse({"message": {"content": "none"}}, status_code=500)

    monkeypatch.setattr(on.requests, "post", fake_post)

    with pytest.raises(requests.HTTPError):
        on.normalize_with_llm("bad response", api_url="http://api")
