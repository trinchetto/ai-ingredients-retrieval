from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from ai_ingredients import dataset_normalizer as dn


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (["tomato", 42], ["tomato", "42"]),
        ('["egg", "milk"]', ["egg", "milk"]),
        ("'salt'", ["salt"]),
        ("", []),
    ],
)
def test_load_cleaned_ingredients_coerces_supported_values(
    tmp_path: Path, raw_value: object, expected: list[str]
) -> None:
    """Ensure load_cleaned_ingredients handles the supported column encodings."""
    frame = pd.DataFrame({"Cleaned_Ingredients": [raw_value]})
    csv_path = tmp_path / "ingredients.csv"
    frame.to_csv(csv_path, index=False)

    result = dn.load_cleaned_ingredients(csv_path)

    assert result == [expected]


def test_load_cleaned_ingredients_raises_for_missing_column(tmp_path: Path) -> None:
    """Expect a ValueError when the requested column does not exist."""
    frame = pd.DataFrame({"OtherColumn": [["salt", "pepper"]]})
    csv_path = tmp_path / "ingredients.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing.*Cleaned_Ingredients"):
        dn.load_cleaned_ingredients(csv_path)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (["basil", "mint"], ["basil", "mint"]),
        ('["thyme", "sage"]', ["thyme", "sage"]),
        ("rosemary", ["rosemary"]),
        ("", []),
        (123, []),
    ],
)
def test_coerce_to_list_supports_multiple_representations(
    input_value: object, expected: list[str]
) -> None:
    """Check that _coerce_to_list normalizes the supported value types."""
    assert dn._coerce_to_list(input_value) == expected


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([["tomato ", "", "pepper"]], [["norm_pre_tomato", "norm_pre_pepper"]]),
        ([["onion"], ["garlic", ""]], [["norm_pre_onion"], ["norm_pre_garlic"]]),
    ],
)
def test_normalize_ingredient_lists_processes_each_item(
    rows: Iterable[list[str]],
    expected: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify normalize_ingredient_lists cleans and normalizes every non-empty item."""

    def fake_preclean(value: str) -> str:
        return f"pre_{value.strip()}"

    def fake_normalize(value: str, **_: object) -> str:
        return f"norm_{value}"

    def passthrough_tqdm(iterable: Iterable[list[str]], **_: object):
        return iterable

    monkeypatch.setattr(dn, "preclean", fake_preclean)
    monkeypatch.setattr(dn, "normalize_with_llm", fake_normalize)
    monkeypatch.setattr(dn, "tqdm", passthrough_tqdm)

    result = dn.normalize_ingredient_lists(
        rows,
        model="dummy",
        temperature=0.0,
        num_predict=1,
        api_url="http://example.com",
        timeout=1.0,
    )

    assert result == expected
