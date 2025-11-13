from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from ai_ingredients import ingredient_clusterer as ic


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([[" salt ", "pepper", "salt"]], ["salt", "pepper"]),
        ([["apple"], ["apple ", "banana"]], ["apple", "banana"]),
        ([["", None, "ginger"]], ["ginger"]),
    ],
)
def test_load_ingredients_produces_unique_trimmed_entries(
    tmp_path: Path,
    payload: list[list[object]],
    expected: list[str],
) -> None:
    """Ensure load_ingredients flattens, trims, and deduplicates rows."""
    path = tmp_path / "normalized.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert ic.load_ingredients(path) == expected


def test_load_ingredients_raises_for_malformed_json(tmp_path: Path) -> None:
    """Expect JSON decoding failures to propagate to the caller."""
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ic.load_ingredients(path)


@pytest.mark.parametrize(
    ("texts", "requested_clusters"),
    [
        (["apple", "apples", "banana"], 2),
        (["salt", "pepper", "paprika", "basil"], 3),
    ],
)
def test_cluster_ingredients_groups_all_members(
    texts: Sequence[str],
    requested_clusters: int,
) -> None:
    """Cluster small corpora and confirm every ingredient is assigned."""
    clusters = ic.cluster_ingredients(
        texts,
        clusters=requested_clusters,
        seed=0,
        niter=20,
    )

    flattened = sorted(member for cluster in clusters for member in cluster["members"])
    assert flattened == sorted(set(texts))
    assert 1 <= len(clusters) <= requested_clusters
    for cluster in clusters:
        assert cluster["centroid"] in cluster["members"]


def test_cluster_ingredients_raises_for_non_positive_clusters() -> None:
    """Validate clusters parameter must be strictly positive."""
    with pytest.raises(ValueError, match="positive"):
        ic.cluster_ingredients(["salt"], clusters=0)


def test_cluster_ingredients_raises_when_clusters_exceed_texts() -> None:
    """Requesting more clusters than data points should fail fast."""
    with pytest.raises(ValueError, match="cannot exceed"):
        ic.cluster_ingredients(["salt"], clusters=2)


def test_cluster_ingredients_raises_when_no_texts_available() -> None:
    """Empty corpora should raise because clusters would be invalid."""
    with pytest.raises(ValueError):
        ic.cluster_ingredients([], clusters=1)


def test_write_clusters_outputs_jsonl(tmp_path: Path) -> None:
    """write_clusters emits one JSON object per line."""
    destination = tmp_path / "clusters.jsonl"
    clusters: list[ic.Cluster] = [
        ic.Cluster(centroid="salt", members=["salt", "sea salt"])
    ]

    ic.write_clusters(clusters, destination)

    content = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    assert json.loads(content[0]) == clusters[0]
