"""Retrieval purity: do nearest neighbours share the query's class?

One matmul over L2-normalised embeddings. No index, no FAISS -- at 3.4k x 512
brute force is milliseconds and is what IndexFlatIP does anyway.

The same-image exclusion is the load-bearing part. Two aircraft parked side by
side in one tile are near-duplicates; counting them as neighbours measures
tile co-occurrence, not class semantics.
"""

from __future__ import annotations

import numpy as np

TOP_K = 5
SEED = 42

# COCO-standard area edges, in px^2.
SIZE_EDGES = (32 * 32, 96 * 96)
SIZE_LABELS = ("<32^2", "32^2-96^2", ">96^2")

# Object-crop upsample factor: 224 / object window side.
UPSAMPLE_EDGES = (2.0, 4.0, 6.0)
UPSAMPLE_LABELS = ("<=2x", "2-4x", "4-6x", ">6x")


def bucket(value: float, edges, labels) -> str:
    for label, edge in zip(labels, edges):
        if value < edge:
            return label
    return labels[-1]


def size_bucket(area_px: float) -> str:
    return bucket(area_px, SIZE_EDGES, SIZE_LABELS)


def upsample_bucket(factor: float) -> str:
    return bucket(factor, UPSAMPLE_EDGES, UPSAMPLE_LABELS)


def tertile_edges(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [1 / 3, 2 / 3])
    return float(lower), float(upper)


def eligibility_mask(
    query_image_codes: np.ndarray,
    db_image_codes: np.ndarray | None = None,
    query_groups: np.ndarray | None = None,
    db_groups: np.ndarray | None = None,
) -> np.ndarray:
    """True where a database crop may serve as a neighbour of a query.

    Excludes every crop from the query's own source image -- which, when the
    query set is the database, also excludes the query itself, and when the
    query is a jittered or background crop, excludes its own GT counterpart
    and everything else in that tile.

    `*_groups` additionally restricts candidates to those sharing a group
    code with the query -- used to ask "how good is retrieval once the sensor
    shortcut is taken away".
    """
    db_image_codes = query_image_codes if db_image_codes is None else db_image_codes
    eligible = query_image_codes[:, None] != db_image_codes[None, :]
    if query_groups is not None:
        db_groups = query_groups if db_groups is None else db_groups
        eligible &= query_groups[:, None] == db_groups[None, :]
    return eligible


def retrieve(
    query_embeddings: np.ndarray,
    query_image_codes: np.ndarray,
    db_embeddings: np.ndarray | None = None,
    db_image_codes: np.ndarray | None = None,
    top_k: int = TOP_K,
    query_groups: np.ndarray | None = None,
    db_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank the database against each query.

    Returns (neighbour indices [n, top_k], their similarities [n, top_k],
    eligible count per query [n]). The database defaults to the query set,
    which is the clean-corpus self-retrieval case.
    """
    db_embeddings = query_embeddings if db_embeddings is None else db_embeddings
    db_image_codes = query_image_codes if db_image_codes is None else db_image_codes
    if query_embeddings.shape[0] != query_image_codes.shape[0]:
        raise ValueError("query embeddings and image_codes disagree on row count")
    if db_embeddings.shape[0] != db_image_codes.shape[0]:
        raise ValueError("database embeddings and image_codes disagree on row count")

    eligible = eligibility_mask(query_image_codes, db_image_codes, query_groups, db_groups)
    neighbours, scores = topk_from_similarity(
        query_embeddings @ db_embeddings.T, eligible, top_k
    )
    return neighbours, scores, eligible.sum(axis=1)


def topk_from_similarity(
    similarity: np.ndarray,
    eligible: np.ndarray | None,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k columns per row of an arbitrary similarity matrix.

    Split out from `retrieve` so a fused score -- alpha * cos(local) +
    (1 - alpha) * cos(regional) -- goes through exactly the same ranking and
    tie-breaking path as a single-stream score.
    """
    if eligible is not None:
        counts = eligible.sum(axis=1)
        if counts.min() < top_k:
            raise ValueError(
                f"A query has only {counts.min()} eligible neighbours, "
                f"fewer than top_k={top_k}"
            )
        similarity = np.where(eligible, similarity, -np.inf)
    elif similarity.shape[1] < top_k:
        raise ValueError(f"Only {similarity.shape[1]} candidates, fewer than top_k={top_k}")

    # argpartition for the top-k, then sort just those k.
    partition = np.argpartition(-similarity, top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(similarity, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1, kind="stable")
    neighbours = np.take_along_axis(partition, order, axis=1)
    return neighbours, np.take_along_axis(similarity, neighbours, axis=1)


def purity(
    neighbours: np.ndarray,
    query_class_ids: np.ndarray,
    db_class_ids: np.ndarray | None = None,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_at_1 per query, p_at_k per query)."""
    db_class_ids = query_class_ids if db_class_ids is None else db_class_ids
    matches = db_class_ids[neighbours] == query_class_ids[:, None]
    return matches[:, 0].astype(np.float64), matches[:, :top_k].mean(axis=1)


def random_baseline(
    query_image_codes: np.ndarray,
    query_class_ids: np.ndarray,
    db_image_codes: np.ndarray | None = None,
    db_class_ids: np.ndarray | None = None,
    top_k: int = TOP_K,
    seed: int = SEED,
    query_groups: np.ndarray | None = None,
    db_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Empirical chance floor: draw top_k random eligible neighbours per query.

    Drawn rather than computed analytically, so it carries the same
    same-image exclusion and the same finite-sample noise as the real measurement.
    """
    db_image_codes = query_image_codes if db_image_codes is None else db_image_codes
    db_class_ids = query_class_ids if db_class_ids is None else db_class_ids
    rng = np.random.default_rng(seed)
    eligible = eligibility_mask(query_image_codes, db_image_codes, query_groups, db_groups)
    scores = np.empty(len(query_class_ids), dtype=np.float64)
    for query in range(len(query_class_ids)):
        candidates = np.flatnonzero(eligible[query])
        drawn = rng.choice(candidates, size=top_k, replace=False)
        scores[query] = np.mean(db_class_ids[drawn] == query_class_ids[query])
    return scores


def auroc(object_scores: np.ndarray, background_scores: np.ndarray) -> float:
    """P(a random real object scores above a random background crop).

    0.5 means the score carries no information about whether a crop contains
    an object; 1.0 means a threshold separates them perfectly.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.ones(len(object_scores)), np.zeros(len(background_scores))])
    return float(roc_auc_score(labels, np.concatenate([object_scores, background_scores])))


def rejection_threshold(
    object_scores: np.ndarray,
    background_scores: np.ndarray,
    background_reject_rate: float = 0.95,
) -> tuple[float, float]:
    """Threshold rejecting `background_reject_rate` of background, and its cost.

    Rejection means sim@1 < threshold. Returns (threshold, fraction of real
    objects also rejected) -- the false-reject cost of buying that much
    background rejection.
    """
    threshold = float(np.quantile(background_scores, background_reject_rate))
    return threshold, float((object_scores < threshold).mean())


def group_mean(values: np.ndarray, keys: np.ndarray) -> dict:
    """Mean of `values` per distinct key, with counts."""
    return {
        key: (float(values[keys == key].mean()), int((keys == key).sum()))
        for key in sorted(set(keys.tolist()))
    }


def _self_check() -> None:
    """Two synthetic corpora with known answers."""
    # Six images, two crops each, two classes; embeddings make class the only
    # signal. Every query then has 4 same-class candidates outside its own
    # image, so a top-4 query is satisfiable without falling back to the
    # other class -- top_k must stay <= that or perfect purity is unreachable
    # by construction rather than by the embedding being bad.
    class_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    image_codes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    basis = np.eye(2, dtype=np.float32)
    embeddings = basis[class_ids]

    neighbours, scores, eligible_counts = retrieve(embeddings, image_codes, top_k=4)
    # 12 crops minus the 2 sharing the query's image.
    assert set(eligible_counts.tolist()) == {10}, eligible_counts
    # No neighbour may come from the query's own image.
    for query, row in enumerate(neighbours):
        assert image_codes[query] not in image_codes[row], (query, row)
    p1, p4 = purity(neighbours, class_ids, top_k=4)
    assert p1.mean() == 1.0 and p4.mean() == 1.0, (p1.mean(), p4.mean())
    # Scores are the similarities of the returned neighbours, descending.
    assert np.allclose(scores, 1.0), scores
    assert np.all(np.diff(scores, axis=1) <= 1e-6)

    # Cross-set: queries are perturbed copies, database is the clean corpus.
    # The query's own image is still excluded, so a query cannot retrieve the
    # crop it was derived from.
    queries = embeddings + 0.01
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    neighbours, _, counts = retrieve(
        queries, image_codes, embeddings, image_codes, top_k=4
    )
    assert set(counts.tolist()) == {10}, counts
    for query, row in enumerate(neighbours):
        assert image_codes[query] not in image_codes[row], (query, row)
    _, p4 = purity(neighbours, class_ids, class_ids, top_k=4)
    assert p4.mean() == 1.0, p4.mean()

    # Same corpus, but every neighbour drawn from the query's own image would
    # be a perfect match: purity must NOT see them.
    leaky = np.zeros((12, 2), dtype=np.float32)
    leaky[:, 0] = 1.0
    for pair in range(0, 12, 2):
        leaky[pair : pair + 2] = basis[1] if pair % 4 else basis[0]
    neighbours, _, _ = retrieve(leaky, image_codes, top_k=4)
    for query, row in enumerate(neighbours):
        assert image_codes[query] not in image_codes[row], (query, row)

    # Degenerate corpus: identical embeddings, so ranking is arbitrary but the
    # exclusion must still hold.
    flat = np.ones((12, 2), dtype=np.float32) / np.sqrt(2)
    neighbours, _, _ = retrieve(flat, image_codes, top_k=4)
    for query, row in enumerate(neighbours):
        assert image_codes[query] not in image_codes[row]

    baseline = random_baseline(image_codes, class_ids, top_k=4)
    assert 0.0 <= baseline.mean() <= 1.0

    # Novelty metrics. Perfect separation, then none.
    high, low = np.array([0.9, 0.8, 0.85]), np.array([0.2, 0.3, 0.1])
    assert auroc(high, low) == 1.0
    assert auroc(low, high) == 0.0
    assert abs(auroc(high, high) - 0.5) < 1e-9
    # Threshold at 95% background rejection sits above almost all background.
    threshold, cost = rejection_threshold(high, low, 0.95)
    assert 0.29 <= threshold <= 0.31 and cost == 0.0, (threshold, cost)
    # When the two overlap completely, buying 95% rejection costs nearly all
    # the real objects too -- the failure mode this metric exists to expose.
    overlapping = np.linspace(0.1, 0.9, 100)
    _, cost = rejection_threshold(overlapping, overlapping, 0.95)
    assert cost >= 0.9, cost

    assert size_bucket(100) == "<32^2"
    assert size_bucket(5000) == "32^2-96^2"
    assert size_bucket(50000) == ">96^2"
    assert upsample_bucket(1.5) == "<=2x" and upsample_bucket(8.0) == ">6x"

    print("gate self-check OK")


if __name__ == "__main__":
    _self_check()
