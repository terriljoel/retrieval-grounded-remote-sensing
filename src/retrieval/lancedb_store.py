from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.annotation.models import Box
from src.retrieval.crops import crop_image
from src.retrieval.models import EvidenceRecord


REQUIRED_COLUMNS = {
    "embedding_id",
    "image_id",
    "image_key",
    "dataset_image_path",
    "split",
    "record_type",
    "crop_type",
    "class_id",
    "class_name",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "context_margin",
    "verified",
    "eligible_as_evidence",
    "vector",
}


def sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def find_latest_database(artifact_root: Path) -> Path:
    artifact_root = Path(artifact_root).resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Retrieval artifact root not found: {artifact_root}")
    candidates = [
        path / "lancedb"
        for path in artifact_root.iterdir()
        if path.is_dir() and (path / "lancedb").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No versioned LanceDB artifact found under: {artifact_root}"
        )
    return max(candidates, key=lambda path: path.parent.stat().st_mtime)


class LanceDbEvidenceStore:
    def __init__(
        self,
        *,
        database_path: Path,
        table_name: str,
        dataset_root: Path,
    ):
        try:
            import lancedb
        except ImportError as error:
            raise RuntimeError(
                "LanceDB is required. Install the annotation dependencies."
            ) from error
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_dir():
            raise FileNotFoundError(f"LanceDB directory not found: {self.database_path}")
        self.dataset_root = Path(dataset_root).resolve()
        self.database = lancedb.connect(str(self.database_path))
        if table_name not in self.database.table_names():
            raise ValueError(
                f"LanceDB table {table_name!r} not found. Available: "
                f"{self.database.table_names()}"
            )
        self.table = self.database.open_table(table_name)
        missing = REQUIRED_COLUMNS - set(self.table.schema.names)
        if missing:
            raise ValueError(f"LanceDB table is missing columns: {sorted(missing)}")

    def resolve_image_path(self, dataset_image_path: str) -> Path:
        stored = Path(dataset_image_path)
        path = stored if stored.is_absolute() else self.dataset_root / stored
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Evidence source image not found: {path}")
        return path

    def search(
        self,
        vector: np.ndarray,
        *,
        crop_type: str,
        evidence_splits: Iterable[str],
        top_k: int,
        candidate_limit: int,
        exclude_image_id: str | None = None,
    ) -> list[EvidenceRecord]:
        split_values = ", ".join(sql_literal(split) for split in evidence_splits)
        if not split_values:
            raise ValueError("At least one evidence split is required")
        filters = [
            "verified = true",
            "eligible_as_evidence = true",
            f"split IN ({split_values})",
            "record_type = 'object'",
            f"crop_type = {sql_literal(crop_type)}",
        ]
        if exclude_image_id:
            filters.append(f"image_id != {sql_literal(exclude_image_id)}")
        candidates = (
            self.table.search(np.asarray(vector, dtype=np.float32))
            .distance_type("cosine")
            .where(" AND ".join(filters), prefilter=True)
            .select([
                "embedding_id",
                "image_id",
                "image_key",
                "dataset_image_path",
                "split",
                "class_id",
                "class_name",
                "crop_type",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "context_margin",
                "_distance",
            ])
            .limit(max(candidate_limit, top_k))
            .to_pandas()
        )
        if candidates.empty:
            return []
        candidates = (
            candidates.sort_values("_distance")
            .drop_duplicates(subset="image_id", keep="first")
            .head(top_k)
        )
        results: list[EvidenceRecord] = []
        for row in candidates.to_dict(orient="records"):
            results.append(EvidenceRecord(
                embedding_id=str(row["embedding_id"]),
                image_id=str(row["image_id"]),
                image_key=str(row["image_key"]),
                source_path=self.resolve_image_path(str(row["dataset_image_path"])),
                split=str(row["split"]),
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]),
                crop_type=str(row["crop_type"]),
                box=Box(
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                ),
                context_margin=float(row.get("context_margin") or 0.0),
                cosine_similarity=1.0 - float(row["_distance"]),
            ))
        return results

    @staticmethod
    def evidence_image(record: EvidenceRecord) -> Image.Image:
        """Load the complete source image for a retrieved evidence record."""
        with Image.open(record.source_path) as opened:
            return opened.convert("RGB")

    @staticmethod
    def evidence_crop(record: EvidenceRecord) -> Image.Image:
        image = LanceDbEvidenceStore.evidence_image(record)
        margin = record.context_margin if record.crop_type == "context" else 0.0
        return crop_image(image, record.box, margin_fraction=margin)
