"""Shared Phase 1 setup: corpora, embeddings, proposals, train/val split.

The α-sweep, the open-set test and the baselines all need the identical
corpora and the identical synthetic proposal set. Three copies of that block
would be three places for the seed, the IoU threshold or the memory/query
split to drift apart, and a drift there is invisible -- the numbers would just
quietly stop being comparable.

Everything here is cached on disk, so building a second Setup in a later
script costs a few seconds, not a re-embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.retrieval.crops import extract_crops, load_scope
from src.retrieval.embed import embed_stream
from src.retrieval.perturb import LOW_IOU_BANDS, build_background_corpus, build_jitter_corpus
from src.retrieval.proposals import build_proposals, confusion_distribution
from src.retrieval.sensor import sensor_table

SEED = 42
ENCODER = "remoteclip_l14"
SPLIT_CSV = "configs/splits/nwpu_vhr10_multilabel_split_seed42_fixed.csv"
DATASET_ROOT = "docs/shared_resources/datasets/raw"
BACKGROUND_FROM_NEGATIVES = 500
BACKGROUND_FROM_POSITIVES = 500
MAIN_SCALE = 4.0
ARMS = ("local", "local_tight")


def spaced(items: list, limit: int | None) -> list:
    """A limit that samples across the listing rather than truncating it.

    Truncating would take the first N images, which in NWPU means the first
    one or two classes -- a smoke run that exercises a tenth of the code.
    """
    if limit is None or limit >= len(items):
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


@dataclass
class Setup:
    root: Path
    encoder: str
    scale: float
    index: dict[str, list[dict]]
    proposals: list[dict]
    train: list[dict]
    val: list[dict]
    class_names: list[str]
    confusable: dict
    _row_of: dict = field(repr=False, default_factory=dict)
    _cache: dict = field(repr=False, default_factory=dict)
    _force: bool = False

    @property
    def regional(self) -> str:
        return f"regional_k{self.scale:g}"

    @property
    def streams(self) -> list[str]:
        return list(ARMS) + [self.regional]

    def embeddings(self, corpus: str, stream: str) -> np.ndarray:
        if (corpus, stream) not in self._cache:
            self._cache[corpus, stream] = embed_stream(
                [entry["crop_id"] for entry in self.index[corpus]],
                self.root / "data" / "crops" / corpus,
                self.root / "data" / "embeddings" / corpus,
                self.encoder, stream, force=self._force,
            )
        return self._cache[corpus, stream]

    def gather(self, records: list[dict], stream: str) -> np.ndarray:
        """Embeddings for a proposal list, in the order given."""
        return np.stack([
            self.embeddings(record["corpus"], stream)[
                self._row_of[record["corpus"], record["crop_id"]]
            ]
            for record in records
        ])


def build_corpora(
    root: Path,
    scale: float,
    force: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    positives = spaced(load_scope(root / SPLIT_CSV, root / DATASET_ROOT), limit)
    negatives = spaced(load_scope(root / SPLIT_CSV, root / DATASET_ROOT, background=True), limit)
    sensors = {
        uid: record["sensor"] for uid, record in sensor_table({
            record["image_uid"]: record["image_path"] for record in positives + negatives
        }).items()
    }
    scales = (scale,)
    clean = [
        entry for entry in
        extract_crops(positives, root / "data/crops/clean", scales, force, sensors)
        if not entry["skipped_reason"]
    ]
    jitter = build_jitter_corpus(
        positives, root / "data/crops/jitter", sensors, SEED, scales, force
    )
    # A second, worse-localised corpus. The default jitter spans IoU 0.4-0.7,
    # which is the whole `adjust` range, so without this `reject_localisation`
    # would have no members at all.
    jitter_low = build_jitter_corpus(
        positives, root / "data/crops/jitter_low", sensors, SEED + 1, scales, force,
        bands=LOW_IOU_BANDS,
    )
    sizes = [(e["xmax"] - e["xmin"], e["ymax"] - e["ymin"]) for e in clean]
    background = build_background_corpus(
        negatives, positives, sizes, root / "data/crops/background", sensors, SEED,
        BACKGROUND_FROM_NEGATIVES, BACKGROUND_FROM_POSITIVES, scales, force,
    )
    if verbose:
        print(f"  clean {len(clean)}  jitter {len(jitter)}  "
              f"jitter_low {len(jitter_low)}  background {len(background)}")
    return {"clean": clean, "jitter": jitter,
            "jitter_low": jitter_low, "background": background}, sensors


def prepare(
    root: Path,
    encoder: str = ENCODER,
    scale: float = MAIN_SCALE,
    force: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> Setup:
    if verbose:
        print(f"Building corpora at k={scale:g}")
    index, _ = build_corpora(root, scale, force, limit, verbose)

    setup = Setup(
        root=root, encoder=encoder, scale=scale, index=index,
        proposals=[], train=[], val=[], class_names=[], confusable={},
        _row_of={
            (corpus, entry["crop_id"]): row
            for corpus, entries in index.items()
            for row, entry in enumerate(entries)
        },
        _force=force,
    )

    setup.class_names = sorted({entry["class_name"] for entry in index["clean"]})
    setup.confusable = confusion_distribution(
        root / "reports/gate/confusion_pairs.csv", setup.class_names,
        setup.embeddings("clean", "local"),
        np.array([entry["class_name"] for entry in index["clean"]]),
    )
    setup.proposals = build_proposals(
        index["clean"], index["jitter"], index["background"], setup.confusable, SEED,
        jitter_low_index=index["jitter_low"],
    )
    setup.train = [r for r in setup.proposals if r["split"] == "train"]
    setup.val = [r for r in setup.proposals if r["split"] == "val"]

    # Memory and queries must not share an image, or every query would find
    # its own tile in the memory and every number here would be meaningless.
    overlap = {r["image_uid"] for r in setup.train} & {r["image_uid"] for r in setup.val}
    if overlap:
        raise ValueError(f"{len(overlap)} images appear in both memory and query sets")
    return setup


def selected_alpha(summary_csv: Path, arm: str = "local", default: float = 0.5) -> float:
    """The alpha the sweep selected, read back from its own summary.

    Selection is on `bg95_accept_cost` -- the operating point -- never on
    AUROC. Reading it from the CSV rather than hard-coding a number keeps that
    rule in one place: if the sweep is rerun and the winner moves, every
    downstream script follows without being edited.
    """
    import csv as _csv

    if not summary_csv.is_file():
        return default
    with summary_csv.open(encoding="utf-8", newline="") as file:
        rows = [row for row in _csv.DictReader(file) if row["arm"] == arm]
    if not rows:
        return default
    return float(min(rows, key=lambda row: float(row["bg95_accept_cost"]))["alpha"])


def _self_check() -> None:
    # Pure-logic check: no dataset needed, so this stays runnable anywhere.
    assert spaced(list(range(10)), None) == list(range(10))
    assert spaced(list(range(10)), 20) == list(range(10))
    # A limit must span the listing, not truncate to its head -- NWPU is laid
    # out by class, so a truncating limit would smoke-test one class.
    sampled = spaced(list(range(100)), 5)
    assert sampled == [0, 20, 40, 60, 80], sampled
    assert len(sampled) == 5 and sampled[-1] > 50

    setup = Setup(root=Path("."), encoder="e", scale=4.0, index={},
                  proposals=[], train=[], val=[], class_names=[], confusable={})
    assert setup.regional == "regional_k4"
    assert setup.streams == ["local", "local_tight", "regional_k4"]
    assert Setup(root=Path("."), encoder="e", scale=2.0, index={}, proposals=[],
                 train=[], val=[], class_names=[], confusable={}).regional == "regional_k2"

    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "s.csv"
        path.write_text(
            "arm,alpha,bg95_accept_cost,macro_auroc\n"
            "local,0.75,0.0024,0.889\n"
            "local,1.0,0.0902,0.891\n"          # best AUROC, 37x worse in practice
            "local_tight,0.5,0.0001,0.5\n"      # other arm must not be considered
        )
        assert selected_alpha(path) == 0.75, "selection must minimise cost, not maximise AUROC"
        assert selected_alpha(Path(directory) / "missing.csv") == 0.5
    print("phase1 self-check OK")


if __name__ == "__main__":
    _self_check()
