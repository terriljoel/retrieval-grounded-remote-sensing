"""The verified-case memory: cases.db (SQLite) + embeddings.npy, joined on row index.

Two files, deliberately. The embeddings are a dense float32 matrix that wants
to be memory-mapped and multiplied; the case records are sparse, typed and
queried by predicate. Putting the vectors in SQLite would make the matmul
impossible, and putting the metadata in numpy would make `WHERE decision =
'accept'` a scan.

embeddings.npy is (n_cases, n_streams, dim). The stream order lives in the
`meta` table, not in a sidecar file, because the DB is the thing that is
already being joined against.

`timestamp` is mandatory and carries a seeded synthetic annotation order --
the memory-growth curve replays cases in timestamp order, so a memory built
in class order (which is how NWPU is laid out on disk) would produce a growth
curve that is an artifact of the file listing.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.retrieval.proposals import POSITIVE_VERDICT

ANNOTATORS = ("ann_01", "ann_02", "ann_03")
EPOCH = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)   # a Monday morning
SECONDS_PER_CASE = 45                                      # plausible verification pace

SCHEMA = """
CREATE TABLE cases (
    case_id             INTEGER PRIMARY KEY,
    embedding_row_idx   INTEGER NOT NULL UNIQUE,
    proposal_id         TEXT    NOT NULL UNIQUE,
    image_uid           TEXT    NOT NULL,
    split               TEXT    NOT NULL,
    xmin INTEGER NOT NULL, ymin INTEGER NOT NULL,
    xmax INTEGER NOT NULL, ymax INTEGER NOT NULL,
    dataset             TEXT    NOT NULL,
    predicted_class     TEXT    NOT NULL,
    verified_class      TEXT,
    detector_confidence REAL,
    decision            TEXT    NOT NULL,
    provenance          TEXT    NOT NULL,
    sensor              TEXT    NOT NULL,
    timestamp           TEXT    NOT NULL,
    annotator_id        TEXT    NOT NULL,
    iou                 REAL,
    area_px             REAL,
    upsample_factor     REAL,
    box_fill            REAL
);
CREATE INDEX idx_decision  ON cases(decision);
CREATE INDEX idx_dataset   ON cases(dataset);
CREATE INDEX idx_timestamp ON cases(timestamp);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

CASE_COLUMNS = (
    "embedding_row_idx", "proposal_id", "image_uid", "split",
    "xmin", "ymin", "xmax", "ymax", "dataset", "predicted_class", "verified_class",
    "detector_confidence", "decision", "provenance", "sensor", "timestamp",
    "annotator_id", "iou", "area_px", "upsample_factor", "box_fill",
)


def annotation_order(proposals: list[dict], seed: int) -> list[int]:
    """A seeded arrival order for synthetic cases.

    Random, not corpus order: NWPU is laid out by class, so replaying cases in
    index order would grow the memory one class at a time and the growth curve
    would measure the file listing rather than the memory.
    """
    order = np.arange(len(proposals))
    np.random.default_rng(seed).shuffle(order)
    return order.tolist()


def build_memory(
    proposals: list[dict],
    embeddings: dict[str, np.ndarray],
    db_path: Path,
    npy_path: Path,
    encoder_key: str,
    seed: int,
) -> None:
    """Write cases.db and embeddings.npy for the given proposals.

    `embeddings` maps stream name -> (len(proposals), dim), rows already in
    proposal order.
    """
    streams = sorted(embeddings)
    stacked = np.stack([embeddings[stream] for stream in streams], axis=1)
    if stacked.shape[0] != len(proposals):
        raise ValueError(f"{stacked.shape[0]} embedding rows for {len(proposals)} proposals")
    norms = np.linalg.norm(stacked, axis=2)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("memory embeddings are not L2-normalised")

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, stacked.astype(np.float32))

    db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(SCHEMA)
        order = annotation_order(proposals, seed)
        stamps: dict[int, tuple[str, str]] = {}
        for position, index in enumerate(order):
            moment = EPOCH + timedelta(seconds=SECONDS_PER_CASE * position)
            stamps[index] = (moment.isoformat(), ANNOTATORS[position % len(ANNOTATORS)])

        rows = []
        for index, record in enumerate(proposals):
            timestamp, annotator = stamps[index]
            rows.append((
                index, record["proposal_id"], record["image_uid"], record["split"],
                int(record["xmin"]), int(record["ymin"]),
                int(record["xmax"]), int(record["ymax"]),
                record.get("dataset", "nwpu"),
                record["predicted_class"], record["verified_class"],
                record["detector_confidence"], record["verdict"],
                f"nwpu_{record['corpus']}_{record['split']}",
                record["sensor"], timestamp, annotator,
                record["iou"], record["area_px"],
                record["upsample_factor"], record["box_fill"],
            ))
        connection.executemany(
            f"INSERT INTO cases ({','.join(CASE_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(CASE_COLUMNS))})",
            rows,
        )
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("streams", json.dumps(streams)),
                ("encoder", encoder_key),
                ("seed", str(seed)),
                ("embedding_shape", json.dumps(list(stacked.shape))),
                ("built_at", datetime.now(timezone.utc).isoformat()),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def load_memory(db_path: Path, npy_path: Path) -> tuple[list[dict], np.ndarray, list[str]]:
    """Return (case rows ordered by embedding_row_idx, embeddings, stream order)."""
    embeddings = np.load(npy_path)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row) for row in
            connection.execute("SELECT * FROM cases ORDER BY embedding_row_idx")
        ]
        streams = json.loads(
            connection.execute("SELECT value FROM meta WHERE key='streams'").fetchone()[0]
        )
    finally:
        connection.close()

    if [row["embedding_row_idx"] for row in rows] != list(range(len(rows))):
        raise ValueError("embedding_row_idx is not a contiguous 0..n-1 range")
    if embeddings.shape[0] != len(rows):
        raise ValueError(f"{embeddings.shape[0]} embedding rows for {len(rows)} cases")
    return rows, embeddings, streams


def stream_matrix(embeddings: np.ndarray, streams: list[str], stream: str) -> np.ndarray:
    return embeddings[:, streams.index(stream), :]


def decision_mask(rows: list[dict]) -> np.ndarray:
    """True for cases the human accepted -- the positive evidence pool."""
    return np.array([row["decision"] == POSITIVE_VERDICT for row in rows])


def _self_check() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    proposals = [
        {"proposal_id": f"p{i:03d}", "corpus": "clean", "crop_id": f"c{i:03d}",
         "image_uid": f"img{i:03d}", "split": "train", "sensor": "rgb",
         "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "area_px": 100.0,
         "upsample_factor": 2.0, "box_fill": 0.7, "detector_confidence": None,
         "iou": 1.0, "verified_class": "ship", "predicted_class": "ship",
         "verdict": "accept" if i % 2 else "reject_background"}
        for i in range(20)
    ]
    raw = {
        stream: rng.normal(size=(20, 8)).astype(np.float32)
        for stream in ("local", "regional_k4")
    }
    embeddings = {
        stream: value / np.linalg.norm(value, axis=1, keepdims=True)
        for stream, value in raw.items()
    }

    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "cases.db"
        npy_path = Path(directory) / "embeddings.npy"
        build_memory(proposals, embeddings, db_path, npy_path, "remoteclip_l14", seed=42)
        rows, stacked, streams = load_memory(db_path, npy_path)

        assert len(rows) == 20 and stacked.shape == (20, 2, 8)
        assert streams == ["local", "regional_k4"]
        # The join is positional: row i of the matrix is case i.
        assert np.allclose(stream_matrix(stacked, streams, "local")[3], embeddings["local"][3])
        assert decision_mask(rows).sum() == 10
        assert {row["dataset"] for row in rows} == {"nwpu"}

        # timestamp is mandatory, unique, and not in corpus order -- that is
        # the whole point of it for the growth curve.
        stamps = [row["timestamp"] for row in rows]
        assert all(stamps) and len(set(stamps)) == len(stamps)
        assert stamps != sorted(stamps), "arrival order collapsed to corpus order"
        assert len({row["annotator_id"] for row in rows}) == len(ANNOTATORS)
        # detector_confidence survives as a real NULL, not the string "None".
        assert all(row["detector_confidence"] is None for row in rows)

        # Un-normalised embeddings must be refused, not silently stored.
        try:
            build_memory(proposals, {"local": raw["local"] * 3.0},
                         db_path, npy_path, "remoteclip_l14", seed=42)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted un-normalised embeddings")

    print("memory self-check OK")


if __name__ == "__main__":
    _self_check()
