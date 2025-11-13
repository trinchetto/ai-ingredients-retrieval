"""Dataset-driven normalization pipeline using the Ollama normalizer."""

from __future__ import annotations

import argparse
import json
from ast import literal_eval
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from ai_ingredients.ollama_normalizer import (
    DEFAULT_API_URL,
    normalize_with_llm,
    preclean,
)


def load_cleaned_ingredients(
    csv_path: Path,
    column: str = "Cleaned_Ingredients",
) -> list[list[str]]:
    """Load the target CSV and coerce the selected column into list-of-lists."""
    frame = pd.read_csv(csv_path)
    if column not in frame.columns:
        raise ValueError(f"CSV is missing the {column!r} column.")
    return [_coerce_to_list(value) for value in frame[column].tolist()]


def normalize_ingredient_lists(
    lists: Iterable[list[str]],
    *,
    model: str,
    temperature: float,
    num_predict: int,
    api_url: str,
    timeout: float,
) -> list[list[str]]:
    """Normalize every ingredient in the nested list with the Ollama-backed helper."""
    normalized: list[list[str]] = []
    rows = list(lists)
    for items in tqdm(rows, desc="Normalizing ingredients", unit="row"):
        cleaned_row: list[str] = []
        for item in items:
            if not item:
                continue
            cleaned = preclean(item)
            normalized_item = normalize_with_llm(
                cleaned,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                api_url=api_url,
                timeout=timeout,
            )
            cleaned_row.append(normalized_item)
        normalized.append(cleaned_row)
    return normalized


def _coerce_to_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the Cleaned_Ingredients column via a local Ollama model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "dataset/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
        ),
        help="CSV file containing the Cleaned_Ingredients column.",
    )
    parser.add_argument(
        "--column",
        default="Cleaned_Ingredients",
        help="Name of the column that stores list-like cleaned ingredient data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/normalized_ingredients.json"),
        help="Destination JSON file for the normalized list-of-lists.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Ollama model identifier to pass to normalize_with_llm.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature forwarded to the Ollama options payload.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=16,
        help="Maximum tokens to predict per ingredient.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="HTTP endpoint for the Ollama chat API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout (in seconds) applied to each chat request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingredients = load_cleaned_ingredients(args.dataset, args.column)
    normalized = normalize_ingredient_lists(
        ingredients,
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
        api_url=args.api_url,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    print(
        f"Normalized {sum(len(row) for row in normalized)} ingredients "
        f"across {len(normalized)} rows into {args.output}"
    )


if __name__ == "__main__":
    main()
