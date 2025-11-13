"""Cluster normalized ingredients using FAISS K-means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple, TypedDict

import faiss  # type: ignore[import-untyped]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
from tqdm import tqdm

DEFAULT_NORMALIZED = Path("dataset/normalized_ingredients.json")
DEFAULT_OUTPUT = Path("dataset/ingredient_clusters.jsonl")


class Cluster(TypedDict):
    centroid: str
    members: list[str]


def load_ingredients(path: Path) -> list[str]:
    """Load the nested normalized list and return unique, non-empty entries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    ordered: list[str] = []
    for row in data:
        for item in row:
            if item is None:
                continue
            value = str(item).strip()
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
    return ordered


def _vectorize(texts: Sequence[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return np.ascontiguousarray(matrix.toarray().astype("float32"))


def _train_kmeans(
    features: np.ndarray,
    k: int,
    *,
    niter: int,
    seed: int,
) -> Tuple[faiss.Kmeans, np.ndarray]:
    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=k,
        niter=niter,
        verbose=False,
        seed=seed,
    )
    kmeans.train(features)
    _, assignments = kmeans.index.search(features, 1)
    return kmeans, assignments.reshape(-1)


def _select_representative(
    indices: np.ndarray,
    embedding: np.ndarray,
    centroid: np.ndarray,
) -> int:
    cluster_vectors = embedding[indices]
    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
    winner = indices[int(np.argmin(distances))]
    return int(winner)


def cluster_ingredients(
    texts: Sequence[str],
    *,
    clusters: int,
    niter: int = 25,
    seed: int = 1234,
) -> list[Cluster]:
    """Embed ingredients with TF-IDF n-grams and cluster via FAISS K-means."""
    if clusters <= 0:
        raise ValueError("clusters must be a positive integer.")
    if len(texts) < clusters:
        raise ValueError("clusters cannot exceed the number of available texts.")
    features = _vectorize(texts)
    kmeans, assignments = _train_kmeans(
        features,
        clusters,
        niter=niter,
        seed=seed,
    )

    clustered: list[Cluster] = []
    for cluster_id in range(clusters):
        cluster_indices = np.where(assignments == cluster_id)[0]
        if not len(cluster_indices):
            continue
        representative_idx = _select_representative(
            cluster_indices,
            features,
            kmeans.centroids[cluster_id],
        )
        members = sorted(texts[idx] for idx in cluster_indices)
        clustered.append(
            Cluster(
                centroid=texts[representative_idx],
                members=members,
            )
        )
    return sorted(clustered, key=lambda cluster: cluster["centroid"])


def determine_optimal_clusters(
    texts: Sequence[str],
    *,
    min_clusters: int = 2,
    max_clusters: int = 64,
    niter: int = 25,
    seed: int = 1234,
) -> int:
    """Pick the cluster count that maximizes the silhouette score."""
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be >= min_clusters.")
    if len(texts) <= min_clusters:
        raise ValueError("Not enough texts to evaluate cluster counts.")

    features = _vectorize(texts)
    upper = min(max_clusters, len(texts) - 1)
    if upper < min_clusters:
        raise ValueError("Not enough texts to evaluate cluster counts.")

    best_k = min_clusters
    best_score = -1.0
    for k in tqdm(
        range(min_clusters, upper + 1),
        desc="Evaluating cluster counts",
        unit="k",
        leave=False,
    ):
        _, assignments = _train_kmeans(features, k, niter=niter, seed=seed)
        score = silhouette_score(features, assignments)
        tqdm.write(f"k={k} silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _resolve_cluster_count(
    texts: Sequence[str],
    *,
    clusters: int,
    min_clusters: int,
    auto: bool,
    niter: int,
    seed: int,
) -> int:
    if auto:
        if clusters < max(min_clusters, 2):
            raise ValueError(
                "Auto cluster selection requires --clusters >= min cluster count."
            )
        return determine_optimal_clusters(
            texts,
            min_clusters=max(min_clusters, 2),
            max_clusters=clusters,
            niter=niter,
            seed=seed,
        )
    return clusters


def write_clusters(clusters: Iterable[Cluster], destination: Path) -> None:
    """Persist the clustered data as JSON Lines."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for cluster in clusters:
            handle.write(json.dumps(cluster, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster normalized ingredient strings with FAISS."
    )
    parser.add_argument(
        "--normalized-json",
        type=Path,
        default=DEFAULT_NORMALIZED,
        help="Path to the normalized list-of-lists JSON file.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum clusters to consider during auto selection.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=64,
        help="Number of clusters to target.",
    )
    parser.add_argument(
        "--auto-clusters",
        action="store_true",
        help="Use silhouette score to pick the number of clusters up to --clusters.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=25,
        help="FAISS K-means training iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed forwarded to FAISS for reproducible clusters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSONL file for the clusters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingredients = load_ingredients(args.normalized_json)
    cluster_count = _resolve_cluster_count(
        ingredients,
        clusters=args.clusters,
        min_clusters=args.min_clusters,
        auto=args.auto_clusters,
        niter=args.niter,
        seed=args.seed,
    )
    clusters = cluster_ingredients(
        ingredients,
        clusters=cluster_count,
        niter=args.niter,
        seed=args.seed,
    )
    write_clusters(clusters, args.output)
    print(
        f"Clustered {len(ingredients)} ingredients into {len(clusters)} clusters "
        f"and wrote {args.output}"
    )


if __name__ == "__main__":
    main()
