"""A synthetic proposal set standing in for the detector's export.

Four verdict types, each drawn from a corpus that already exists:

  accept              clean GT boxes -- proposal correct, human accepts
  relabel             jittered boxes at IoU >= 0.5 carrying a confusable class
  adjust              jittered boxes at 0.4 <= IoU < 0.7, right class, loose box
  reject_localisation jitter_low boxes at IoU < 0.4 -- box must be redrawn
  reject_background   background crops -- nothing there at all

**`adjust` is a separate verdict because it is a separate action.** A box at
IoU 0.55 is not discarded; a human drags it into place, which costs a fraction
of drawing one from scratch and nothing at all in label correctness. Folding it
into `reject` costed the cheapest error type at the same rate as a corrupted
label and let it dominate the combined risk curve. The relative cost of an
`adjust` admission is deliberately *not* set here -- it is reported as its own
harm column so the weighting stays a stated assumption in the write-up rather
than a constant buried in this file.

The sources are disjoint by construction: each GT box has exactly one jitter in
each corpus, the two jitter corpora occupy non-overlapping IoU ranges, and the
relabel draw partitions the 0.4-0.7 corpus against `adjust`. Nothing appears
under two verdicts, so the memory never holds one embedding under opposite
decisions.

**Relabels have a ceiling and it should not be forgotten.** Building them on
jittered rather than clean boxes, and drawing the wrong class from the measured
confusion *distribution* rather than its argmax, makes them harder than a
pixel-perfect crop with a swapped label -- but the visual evidence still is not
genuinely ambiguous. A synthetic relabel is a correct object seen slightly off
centre and simply called the wrong thing; a real class error happens where the
imagery itself is confusable. Retrieval will look better here than it will on
detector output, and the true class-error distribution cannot be measured until
the export lands.

When the real export lands, swap this module's output for it -- everything
downstream consumes the proposal records, not the corpora.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERDICTS = ("accept", "relabel", "adjust", "reject_localisation", "reject_background")
POSITIVE_VERDICT = "accept"
RELABEL_FRACTION = 0.30     # of the jitters that are well enough localised
LOCALISATION_IOU = 0.5      # at or above this the box is good enough to carry only a class error
ADJUST_IOU = 0.4            # at or above this the box is dragged into place, not redrawn
CENTROID_TEMPERATURE = 0.05  # softmax sharpness for the no-confusion-data fallback


def confusion_distribution(
    confusion_csv: Path,
    class_names: list[str],
    embeddings: np.ndarray | None = None,
    embedding_classes: np.ndarray | None = None,
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Per class, the distribution over classes it gets confused with.

    A distribution rather than an argmax: taking only the single most-confused
    class makes every relabel of a given class identical, which both
    understates the variety of real class errors and lets a gate learn the
    fixed pairing instead of the evidence.

    Primary source is the measured confusion table. Retrieval is good enough
    that some classes never appear there; those fall back to a softmax over
    centroid cosine similarity, which asks the embedding the same question
    directly rather than via the top-5 it happened to return.
    """
    tally: dict[str, Counter] = defaultdict(Counter)
    if confusion_csv.is_file():
        with confusion_csv.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row["query_class"] != row["neighbour_class"]:
                    tally[row["query_class"]][row["neighbour_class"]] += int(row["count"])

    centroids = None
    if embeddings is not None:
        centroids = {
            other: embeddings[embedding_classes == other].mean(axis=0)
            for other in class_names
        }

    distribution: dict[str, tuple[list[str], np.ndarray]] = {}
    for name in class_names:
        if tally[name]:
            targets = sorted(tally[name])
            weights = np.array([tally[name][target] for target in targets], dtype=np.float64)
        elif centroids is not None:
            targets = [other for other in class_names if other != name]
            own = centroids[name]
            similarity = np.array([
                float(own @ centroids[other]
                      / (np.linalg.norm(own) * np.linalg.norm(centroids[other])))
                for other in targets
            ])
            weights = np.exp((similarity - similarity.max()) / CENTROID_TEMPERATURE)
        else:
            raise ValueError(f"No confusion data for {name!r} and no embeddings to fall back on")
        distribution[name] = (targets, weights / weights.sum())
    return distribution


def build_proposals(
    clean_index: list[dict],
    jitter_index: list[dict],
    background_index: list[dict],
    confusable: dict[str, str],
    seed: int,
    relabel_fraction: float = RELABEL_FRACTION,
    jitter_low_index: list[dict] | None = None,
) -> list[dict]:
    """One record per synthetic proposal, sorted by proposal_id."""
    rng = np.random.default_rng(seed)
    proposals: list[dict] = []

    def base(entry: dict, corpus: str) -> dict:
        return {
            "proposal_id": f"{corpus}:{entry['crop_id']}",
            "corpus": corpus,
            "crop_id": entry["crop_id"],
            "image_uid": entry["image_uid"],
            "split": entry["split"],
            "sensor": entry["sensor"],
            "xmin": entry["xmin"], "ymin": entry["ymin"],
            "xmax": entry["xmax"], "ymax": entry["ymax"],
            "area_px": entry["area_px"],
            "upsample_factor": entry["upsample_factor"],
            "box_fill": entry["box_fill"],
            # Not available for synthetic proposals -- see STATE.md. Carried as
            # an explicit None so the column exists the day the export lands.
            "detector_confidence": None,
        }

    # Every clean box is an accept: the box is right and so is the label.
    for entry in clean_index:
        record = base(entry, "clean")
        record.update(
            iou=1.0,
            verified_class=entry["class_name"],
            predicted_class=entry["class_name"],
            verdict="accept",
        )
        proposals.append(record)

    # The low-IoU corpus is unambiguous: the box has to be redrawn.
    for entry in (jitter_low_index or []):
        if entry["iou"] >= ADJUST_IOU:
            continue
        record = base(entry, "jitter_low")
        record.update(
            iou=entry["iou"],
            verified_class=entry["class_name"],
            predicted_class=entry["class_name"],
            verdict="reject_localisation",
        )
        proposals.append(record)

    # The 0.4-0.7 corpus is the drag-into-place range. A sampled fraction of the
    # boxes good enough to carry only a class error (IoU >= 0.5) become
    # relabels; everything else in the corpus is an `adjust`.
    relabel_pool = [e for e in jitter_index if e["iou"] >= LOCALISATION_IOU]
    chosen = set()
    for entry, is_relabel in zip(relabel_pool, rng.random(len(relabel_pool)) < relabel_fraction):
        if is_relabel:
            chosen.add(entry["crop_id"])

    for entry in jitter_index:
        if entry["crop_id"] in chosen or entry["iou"] < ADJUST_IOU:
            continue
        record = base(entry, "jitter")
        record.update(
            iou=entry["iou"],
            verified_class=entry["class_name"],
            predicted_class=entry["class_name"],
            verdict="adjust",
        )
        proposals.append(record)

    for entry in relabel_pool:
        if entry["crop_id"] not in chosen:
            continue
        true_class = entry["class_name"]
        targets, weights = confusable[true_class]
        record = base(entry, "jitter")
        record.update(
            iou=entry["iou"],
            verified_class=true_class,
            predicted_class=targets[rng.choice(len(targets), p=weights)],
            verdict="relabel",
        )
        proposals.append(record)

    # Background false positives. The detector claimed *something*, so draw a
    # class from the real class prior rather than leaving it blank -- a gate
    # that only ever sees plausible class labels on background is the harder
    # and more realistic test.
    prior = Counter(entry["class_name"] for entry in clean_index)
    classes = sorted(prior)
    weights = np.array([prior[name] for name in classes], dtype=np.float64)
    weights /= weights.sum()
    for entry in background_index:
        record = base(entry, "background")
        record.update(
            iou=0.0,
            verified_class=None,
            predicted_class=classes[rng.choice(len(classes), p=weights)],
            verdict="reject_background",
        )
        proposals.append(record)

    return sorted(proposals, key=lambda record: record["proposal_id"])


def _self_check() -> None:
    clean = [
        {"crop_id": f"c{i:03d}", "image_uid": f"img{i // 3:03d}", "split": "train",
         "sensor": "rgb", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "area_px": 100,
         "upsample_factor": 2.0, "box_fill": 0.7,
         "class_name": ["ship", "vehicle", "airplane"][i % 3]}
        for i in range(60)
    ]
    jitter = [
        {**entry, "iou": 0.45 if index % 2 else 0.65}
        for index, entry in enumerate(clean)
    ]
    jitter_low = [
        {**entry, "iou": 0.10 if index % 2 else 0.30}
        for index, entry in enumerate(clean)
    ]
    background = [
        {"crop_id": f"b{i:03d}", "image_uid": f"neg{i:03d}", "split": "train",
         "sensor": "rgb", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "area_px": 100,
         "upsample_factor": 2.0, "box_fill": 0.7}
        for i in range(20)
    ]
    import numpy as _np
    confusable = {
        "ship": (["vehicle", "airplane"], _np.array([0.7, 0.3])),
        "vehicle": (["ship"], _np.array([1.0])),
        "airplane": (["ship", "vehicle"], _np.array([0.5, 0.5])),
    }

    proposals = build_proposals(clean, jitter, background, confusable, seed=42,
                                relabel_fraction=0.5, jitter_low_index=jitter_low)
    by_verdict = Counter(record["verdict"] for record in proposals)
    assert set(by_verdict) == set(VERDICTS), by_verdict

    # Every clean crop is an accept, exactly once.
    clean_ids = [r["crop_id"] for r in proposals if r["corpus"] == "clean"]
    assert sorted(clean_ids) == sorted(e["crop_id"] for e in clean)
    assert by_verdict["accept"] == len(clean)

    # reject_localisation comes only from the low corpus; nothing at or above
    # the adjust threshold can be a reject.
    assert by_verdict["reject_localisation"] == sum(e["iou"] < ADJUST_IOU for e in jitter_low)
    assert all(r["corpus"] == "jitter_low" for r in proposals
               if r["verdict"] == "reject_localisation")
    assert all(r["iou"] < ADJUST_IOU for r in proposals
               if r["verdict"] == "reject_localisation")

    # adjust and relabel partition the 0.4-0.7 corpus and never overlap.
    assert all(ADJUST_IOU <= r["iou"] for r in proposals if r["verdict"] == "adjust")
    assert all(r["iou"] >= LOCALISATION_IOU for r in proposals if r["verdict"] == "relabel")
    assert by_verdict["adjust"] + by_verdict["relabel"] == sum(
        e["iou"] >= ADJUST_IOU for e in jitter), by_verdict
    jitter_ids = [r["crop_id"] for r in proposals if r["corpus"] == "jitter"]
    assert len(jitter_ids) == len(set(jitter_ids)), "a jitter crop used under two verdicts"
    assert by_verdict["adjust"] > 0 and by_verdict["relabel"] > 0, by_verdict

    # An adjust keeps the true class -- only the box is wrong.
    for record in proposals:
        if record["verdict"] == "adjust":
            assert record["predicted_class"] == record["verified_class"], record

    # Relabels must not all collapse onto one target class -- that is the
    # whole point of sampling the distribution instead of its argmax.
    ship_targets = {r["predicted_class"] for r in proposals
                    if r["verdict"] == "relabel" and r["verified_class"] == "ship"}
    assert len(ship_targets) > 1, ship_targets

    # A relabel must actually carry a different class; an accept must not.
    for record in proposals:
        if record["verdict"] == "relabel":
            assert record["predicted_class"] != record["verified_class"], record
        if record["verdict"] == "accept":
            assert record["predicted_class"] == record["verified_class"], record
        if record["verdict"] == "reject_background":
            assert record["verified_class"] is None and record["predicted_class"]

    # Seeded: same inputs, same proposals, including the sampled class targets.
    again = build_proposals(clean, jitter, background, confusable, seed=42,
                            relabel_fraction=0.5, jitter_low_index=jitter_low)
    assert [r["proposal_id"] for r in again] == [r["proposal_id"] for r in proposals]
    assert [r["predicted_class"] for r in again] == [r["predicted_class"] for r in proposals]
    other = build_proposals(clean, jitter, background, confusable, seed=7,
                            relabel_fraction=0.5, jitter_low_index=jitter_low)
    assert [r["verdict"] for r in other] != [r["verdict"] for r in proposals]

    # Centroid fallback picks the genuinely nearest class when the confusion
    # table has no row for a class.
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    classes = np.array(["ship", "vehicle", "airplane"])
    mapping = confusion_distribution(
        Path("does_not_exist.csv"), ["ship", "vehicle", "airplane"], embeddings, classes
    )
    for name, (targets, weights) in mapping.items():
        assert name not in targets, f"{name} can be relabelled to itself"
        assert abs(weights.sum() - 1.0) < 1e-9
    ship_targets, ship_weights = mapping["ship"]
    assert ship_targets[int(ship_weights.argmax())] == "vehicle", mapping["ship"]

    print("proposals self-check OK")


if __name__ == "__main__":
    _self_check()
