from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
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


def test_resolve_cluster_count_returns_manual_value() -> None:
    """_resolve_cluster_count should return the requested number when auto is off."""
    texts = ["salt", "pepper"]
    assert (
        ic._resolve_cluster_count(
            texts,
            clusters=1,
            min_clusters=2,
            auto=False,
            niter=10,
            seed=0,
        )
        == 1
    )


def test_resolve_cluster_count_uses_silhouette_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode must rely on determine_optimal_clusters for the final value."""
    texts = ["salt", "pepper", "paprika"]

    def fake_determine(
        values: Sequence[str],
        *,
        min_clusters: int,
        max_clusters: int,
        niter: int,
        seed: int,
    ) -> int:
        assert values == texts
        assert min_clusters == 2
        assert max_clusters == 3
        assert niter == 10
        assert seed == 7
        return 2

    monkeypatch.setattr(ic, "determine_optimal_clusters", fake_determine)

    result = ic._resolve_cluster_count(
        texts,
        clusters=3,
        min_clusters=2,
        auto=True,
        niter=10,
        seed=7,
    )

    assert result == 2


def test_resolve_cluster_count_validates_upper_bound() -> None:
    """Auto mode should fail when the supplied max is less than 2."""
    with pytest.raises(ValueError, match="requires --clusters >= min cluster count"):
        ic._resolve_cluster_count(
            ["salt", "pepper"],
            clusters=1,
            min_clusters=2,
            auto=True,
            niter=10,
            seed=0,
        )


def test_determine_optimal_clusters_prefers_highest_silhouette(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pick the cluster count associated with the best silhouette score."""
    texts = ["apple", "apples", "banana", "bananas"]

    def fake_vectorize(_: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype="float32")

    def fake_train(
        features: np.ndarray,
        k: int,
        *,
        niter: int,
        seed: int,
    ):
        assignments_map = {
            2: np.array([0, 0, 1, 1]),
            3: np.array([0, 1, 1, 2]),
        }

        class Dummy:
            def __init__(self, count: int) -> None:
                self.centroids = np.zeros((count, features.shape[1]), dtype="float32")

        return Dummy(k), assignments_map[k]

    def fake_silhouette(_: np.ndarray, labels: np.ndarray) -> float:
        unique = len(set(labels.tolist()))
        return 0.8 if unique == 2 else 0.3

    class FakeTqdm:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def __call__(self, iterable, **_: object):
            return iterable

        def write(self, message: str) -> None:
            self.messages.append(message)

    fake_tqdm = FakeTqdm()

    monkeypatch.setattr(ic, "_vectorize", fake_vectorize)
    monkeypatch.setattr(ic, "_train_kmeans", fake_train)
    monkeypatch.setattr(ic, "silhouette_score", fake_silhouette)
    monkeypatch.setattr(ic, "tqdm", fake_tqdm)

    result = ic.determine_optimal_clusters(
        texts,
        min_clusters=2,
        max_clusters=3,
    )

    assert result == 2
    assert fake_tqdm.messages == ["k=2 silhouette=0.8000", "k=3 silhouette=0.3000"]


def test_determine_optimal_clusters_raises_with_insufficient_texts() -> None:
    """Expect a ValueError when silhouette computation is impossible."""
    with pytest.raises(ValueError):
        ic.determine_optimal_clusters(["salt"], min_clusters=2, max_clusters=3)


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
