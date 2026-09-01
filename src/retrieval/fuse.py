"""Two-scale fused retrieval against the case memory.

    s = alpha * cos(local) + (1 - alpha) * cos(regional)

Embeddings are L2-normalised, so cosine is a dot product and the whole thing
is two matmuls and an add. No index, no FAISS -- at a few thousand cases the
matmul is milliseconds and an index would only add a recall approximation to
apologise for later.

Positive and negative evidence are retrieved *separately* rather than as one
ranked list. A gate needs to know both "how strongly does the memory support
this proposal" and "how strongly does it contradict it", and a single mixed
top-k answers neither: five negative neighbours would look identical to five
positive ones once the class labels are stripped.
"""

from __future__ import annotations

import numpy as np

from src.retrieval.gate import topk_from_similarity

TOP_K = 5


def fused_similarity(
    query_local: np.ndarray,
    query_regional: np.ndarray,
    db_local: np.ndarray,
    db_regional: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 1.0:
        return query_local @ db_local.T
    if alpha == 0.0:
        return query_regional @ db_regional.T
    return alpha * (query_local @ db_local.T) + (1 - alpha) * (query_regional @ db_regional.T)


def evidence(
    similarity: np.ndarray,
    positive_mask: np.ndarray,
    top_k: int = TOP_K,
) -> dict:
    """Split the memory into supporting and contradicting pools and rank each.

    Returns index/score arrays for both pools. Indices are into the *full*
    memory, not into the pool, so they join straight back to the case rows.
    """
    positives = np.flatnonzero(positive_mask)
    negatives = np.flatnonzero(~positive_mask)
    if len(positives) < top_k or len(negatives) < top_k:
        raise ValueError(
            f"memory has {len(positives)} positive and {len(negatives)} negative "
            f"cases; need at least top_k={top_k} of each"
        )

    pos_local, pos_scores = topk_from_similarity(similarity[:, positives], None, top_k)
    neg_local, neg_scores = topk_from_similarity(similarity[:, negatives], None, top_k)
    return {
        "positive_idx": positives[pos_local],
        "positive_sim": pos_scores,
        "negative_idx": negatives[neg_local],
        "negative_sim": neg_scores,
    }


ABSENT_CLASS = -1.0


def class_margin(
    similarity: np.ndarray,
    positive_mask: np.ndarray,
    query_predicted: np.ndarray,
    memory_verified: np.ndarray,
    case_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Best similarity to the predicted class minus best to any other class.

    Taken over the **whole positive pool**, not the top-k. Computing it inside
    a top-5 window made it degenerate: with purity near 1.0 the predicted class
    is either all five neighbours or none of them, so the feature collapsed to
    a sign flip on agreement@k and carried no extra information. Over the full
    memory it is a real margin -- how much better the claimed class is
    supported than its closest rival -- and it stays informative when the two
    are close.

    A predicted class with no accepted case anywhere in the memory scores
    ABSENT_CLASS. That is a genuine "the memory has never seen this class",
    not the old "not in this particular top-5"; open-set queries are meant to
    be caught by novelty, not by this feature.

    `case_weights` (one per memory case) scales each case's support before the
    per-class max, so a down-weighted case has to be *more* similar to win its
    class the same margin. Both the claimed class and its rivals are scaled by
    the same rule, so the comparison stays like-for-like: what changes is that
    a human-verified rival now outranks a machine-accepted claim more easily.
    """
    pool = similarity[:, positive_mask]
    if case_weights is not None:
        pool = pool * case_weights[positive_mask][None, :]
    pool_classes = memory_verified[positive_mask]
    names = sorted(set(pool_classes.tolist()))
    # (n_classes, n_queries): the best support each class offers each query.
    best = np.stack([pool[:, pool_classes == name].max(axis=1) for name in names])

    position = {name: row for row, name in enumerate(names)}
    rows = np.array([position.get(name, -1) for name in query_predicted])
    queries = np.arange(len(query_predicted))

    own = np.where(rows >= 0, best[rows, queries], ABSENT_CLASS)
    # Best over every class except the claimed one: mask it out, then max.
    masked = best.copy()
    masked[rows[rows >= 0], queries[rows >= 0]] = -np.inf
    other = masked.max(axis=0)
    return own - other


def features(
    result: dict,
    query_predicted: np.ndarray,
    memory_verified: np.ndarray,
    top_k: int = TOP_K,
    similarity: np.ndarray | None = None,
    positive_mask: np.ndarray | None = None,
    case_weights: np.ndarray | None = None,
) -> dict:
    """Gate features derived from retrieved evidence.

    `similarity` and `positive_mask` are the ones `evidence` was called with;
    they are needed because `class_margin` is a full-memory statistic.

    `case_weights` is provenance damping: one weight per memory case, applied
    to **only the two class-reading features**, `agreement_at_k` and
    `class_margin`. Ranking, `sim_at_1`, `novelty` and `evidence_margin` stay
    on the unweighted similarity on purpose. Finding 32 measured that label
    corruption attacks the class-reading features and leaves the class-agnostic
    ones almost untouched (background coverage −3.6% at 20% noise), so damping
    those too would trade away a demonstrably robust signal to protect one that
    was never at risk.
    """
    neighbour_classes = memory_verified[result["positive_idx"]]
    agrees = neighbour_classes == query_predicted[:, None]

    sim_at_1 = result["positive_sim"][:, 0]
    if similarity is None or positive_mask is None:
        raise ValueError("class_margin needs the full similarity matrix and positive_mask")
    margin = class_margin(
        similarity, positive_mask, query_predicted, memory_verified, case_weights
    )

    if case_weights is None:
        agreement = agrees[:, :top_k].mean(axis=1)
    else:
        # Weighted vote over the same neighbours: a machine-accepted case still
        # gets retrieved, it just counts for less. Normalised by the weights
        # actually present so the feature stays on [0, 1] and stays comparable
        # to the undamped run.
        neighbour_weights = case_weights[result["positive_idx"][:, :top_k]]
        agreement = ((agrees[:, :top_k] * neighbour_weights).sum(axis=1)
                     / neighbour_weights.sum(axis=1))

    return {
        "sim_at_1": sim_at_1,
        "agreement_at_k": agreement,
        "class_margin": margin,
        "novelty": 1.0 - sim_at_1,
        "negative_sim_at_1": result["negative_sim"][:, 0],
        # Positive evidence minus negative evidence: high when the memory
        # supports the proposal and holds nothing similar that was rejected.
        "evidence_margin": sim_at_1 - result["negative_sim"][:, 0],
    }


def _self_check() -> None:
    # Two classes on orthogonal axes; memory holds accepted cases of both plus
    # rejected cases sitting between them.
    basis = np.eye(3, dtype=np.float32)
    db_local = np.array(
        [basis[0]] * 4 + [basis[1]] * 4 + [(basis[0] + basis[2]) / np.sqrt(2)] * 4
    )
    positive_mask = np.array([True] * 8 + [False] * 4)
    memory_verified = np.array(["ship"] * 4 + ["vehicle"] * 4 + ["ship"] * 4)

    query_local = np.array([basis[0], basis[1]])
    query_regional = np.array([basis[1], basis[0]])      # deliberately swapped
    db_regional = np.array([basis[1]] * 4 + [basis[0]] * 4 + [basis[2]] * 4)

    # alpha = 1 must ignore the regional stream entirely, and vice versa.
    only_local = fused_similarity(query_local, query_regional, db_local, db_regional, 1.0)
    assert np.allclose(only_local, query_local @ db_local.T)
    only_regional = fused_similarity(query_local, query_regional, db_local, db_regional, 0.0)
    assert np.allclose(only_regional, query_regional @ db_regional.T)
    half = fused_similarity(query_local, query_regional, db_local, db_regional, 0.5)
    assert np.allclose(half, 0.5 * only_local + 0.5 * only_regional)

    for bad in (-0.1, 1.1):
        try:
            fused_similarity(query_local, query_regional, db_local, db_regional, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted alpha={bad}")

    result = evidence(only_local, positive_mask, top_k=4)
    # Pools must not leak into each other.
    assert set(result["positive_idx"].ravel().tolist()) <= set(range(8))
    assert set(result["negative_idx"].ravel().tolist()) <= set(range(8, 12))

    computed = features(result, np.array(["ship", "vehicle"]), memory_verified, 4,
                        only_local, positive_mask)
    # Query 0 is a ship and the memory's ships match it exactly.
    assert computed["sim_at_1"][0] == 1.0 and computed["agreement_at_k"][0] == 1.0
    assert computed["novelty"][0] == 0.0
    # Its class margin is positive: ship beats vehicle.
    assert computed["class_margin"][0] > 0

    # A proposal claiming the wrong class must show a negative class margin
    # even though its sim@1 is high -- this is the relabel failure mode.
    wrong = features(result, np.array(["vehicle", "ship"]), memory_verified, 4,
                     only_local, positive_mask)
    assert wrong["class_margin"][0] < 0, wrong["class_margin"]
    assert wrong["sim_at_1"][0] == 1.0, "sim@1 alone cannot see a relabel"

    # The margin must be a real margin, not a sign flip on agreement@k. Ship
    # sits at cos 1.0 from the ship cases and cos 0.0 from the vehicle ones,
    # so the margin is that gap -- not the +-(1 + sim) the sentinel produced.
    assert np.isclose(computed["class_margin"][0], 1.0), computed["class_margin"]
    assert np.isclose(wrong["class_margin"][0], -1.0), wrong["class_margin"]

    # A near-tie must show a small margin even though agreement@k is 1.0:
    # this is the case the top-5 version could not express at all.
    near = np.array([0.99, 0.14, 0.0], dtype=np.float32)
    near /= np.linalg.norm(near)
    tie = features(
        evidence(np.vstack([near]) @ db_local.T, positive_mask, 4),
        np.array(["ship"]), memory_verified, 4,
        np.vstack([near]) @ db_local.T, positive_mask,
    )
    assert tie["agreement_at_k"][0] == 1.0, "top-5 is unanimous"
    assert 0.0 < tie["class_margin"][0] < 0.95, tie["class_margin"]

    # A class the memory has never accepted scores the absent sentinel, and
    # that is the only case the sentinel may fire in.
    unseen = features(result, np.array(["harbor", "ship"]), memory_verified, 4,
                      only_local, positive_mask)
    assert unseen["class_margin"][0] == ABSENT_CLASS - 1.0, unseen["class_margin"]

    # Too few cases in a pool is an error, not a silently short list.
    try:
        evidence(only_local, np.array([True] * 11 + [False]), top_k=4)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a memory with too few negatives")

    # --- provenance damping ------------------------------------------------
    # Memory: 4 ships and 4 vehicles accepted. Query 0 is a ship claiming ship.
    ones = np.ones(12)
    undamped = features(result, np.array(["ship", "vehicle"]), memory_verified,
                        top_k=4, similarity=only_local, positive_mask=positive_mask)
    unit = features(result, np.array(["ship", "vehicle"]), memory_verified, top_k=4,
                    similarity=only_local, positive_mask=positive_mask, case_weights=ones)
    # Weights of exactly 1.0 must be a no-op, or every damped number is
    # incomparable to the run it is supposed to be measured against.
    for name in ("agreement_at_k", "class_margin", "sim_at_1"):
        assert np.allclose(undamped[name], unit[name]), name

    # Down-weight the ships (the claimed class) and nothing else: the margin
    # must fall, because the rival class keeps its full support.
    damped_weights = np.where(memory_verified == "ship", 0.5, 1.0)
    damped = features(result, np.array(["ship", "vehicle"]), memory_verified, top_k=4,
                      similarity=only_local, positive_mask=positive_mask,
                      case_weights=damped_weights)
    assert damped["class_margin"][0] < undamped["class_margin"][0], (
        damped["class_margin"][0], undamped["class_margin"][0])
    # The class-agnostic features must be untouched by any weighting -- that is
    # the whole design decision, so it is asserted rather than trusted.
    for name in ("sim_at_1", "novelty", "evidence_margin", "negative_sim_at_1"):
        assert np.allclose(damped[name], undamped[name]), name

    # A weighted vote stays on [0, 1] and still reads 1.0 when every retrieved
    # neighbour agrees, however hard they are down-weighted.
    tiny = features(result, np.array(["ship", "vehicle"]), memory_verified, top_k=4,
                    similarity=only_local, positive_mask=positive_mask,
                    case_weights=np.full(12, 0.01))
    assert 0.0 <= tiny["agreement_at_k"][0] <= 1.0
    assert np.isclose(tiny["agreement_at_k"][0], undamped["agreement_at_k"][0])

    print("fuse self-check OK")


if __name__ == "__main__":
    _self_check()
