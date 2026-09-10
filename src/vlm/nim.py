from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from src.annotation.models import DetectionSuggestion
from src.retrieval.models import EvidenceRecord


RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class VlmAssessment:
    decision: str
    suggested_class: str | None
    confidence: float | None
    observations: str
    uncertainty: str
    evidence_ids: tuple[str, ...]
    raw_response: str


def encode_image(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def grounded_montage(
    query_image: Image.Image,
    query_crop: Image.Image,
    detection: DetectionSuggestion,
    evidence: list[EvidenceRecord],
    evidence_images: list[Image.Image],
    crops: list[Image.Image],
    tile_size: int = 288,
) -> Image.Image:
    if not (len(evidence) == len(evidence_images) == len(crops)):
        raise ValueError("Evidence records, full images, and crops must be aligned")

    def boxed(image: Image.Image, box, label: str, colour: str) -> Image.Image:
        rendered = image.convert("RGB").copy()
        draw = ImageDraw.Draw(rendered)
        width = max(3, round(min(rendered.size) / 180))
        coordinates = box.as_list()
        draw.rectangle(coordinates, outline=colour, width=width)
        x, y = coordinates[:2]
        text_box = draw.textbbox((x, y), label)
        draw.rectangle(text_box, fill=colour)
        draw.text((x, y), label, fill="black")
        return rendered

    columns = 2
    label_height = 32
    tiles = [
        (
            "Q full: proposed box",
            boxed(query_image, detection.box, detection.class_name, "#00d4ff"),
        ),
        ("Q crop: proposed object", query_crop),
    ]
    for index, (record, source_image, crop) in enumerate(
        zip(evidence, evidence_images, crops, strict=True), start=1
    ):
        tiles.extend([
            (
                f"E{index} full: verified {record.class_name}",
                boxed(source_image, record.box, record.class_name, "#2ec4b6"),
            ),
            (f"E{index} crop: cosine {record.cosine_similarity:.3f}", crop),
        ])
    rows = len(tiles) // columns
    canvas = Image.new(
        "RGB",
        (tile_size * columns, (tile_size + label_height) * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, crop) in enumerate(tiles):
        fitted = crop.convert("RGB").copy()
        fitted.thumbnail((tile_size, tile_size))
        row, column = divmod(index, columns)
        tile_x = column * tile_size
        tile_y = row * (tile_size + label_height)
        x = tile_x + (tile_size - fitted.width) // 2
        y = tile_y + (tile_size - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.text((tile_x + 4, tile_y + tile_size + 6), label, fill="black")
    return canvas


def build_prompt(
    detection: DetectionSuggestion,
    evidence: list[EvidenceRecord],
    classes: Iterable[str],
    mode: str,
) -> str:
    evidence_lines = "\n".join(
        f"E{index}: verified {record.class_name}, cosine similarity "
        f"{record.cosine_similarity:.3f}"
        for index, record in enumerate(evidence, start=1)
    )
    readable_classes = ", ".join(name.replace("_", " ") for name in classes)
    if mode == "query_only":
        panel_description = (
            "Q full is the complete query image with the proposed box marked. "
            "Q crop is the corresponding tight object crop. No retrieved evidence "
            "is supplied in this diagnostic condition."
        )
        evidence_section = "Retrieved evidence: none"
    elif mode == "retrieval_grounded":
        panel_description = (
            "Q full is the complete query image with the proposed box marked and "
            "Q crop is its tight crop. Each E row contains a retrieved verified "
            "example: its complete source image with the verified box marked, then "
            "the corresponding object crop."
        )
        evidence_section = f"Retrieved evidence:\n{evidence_lines}"
    else:
        raise ValueError(f"Unknown VLM assessment mode: {mode}")

    return f"""You assist a human annotating aerial imagery.

{panel_description}

Detector suggestion: {detection.class_name.replace('_', ' ')}
Detector confidence: {detection.confidence:.4f}
Allowed classes: {readable_classes}

{evidence_section}

First judge the proposed object using both its local appearance and the full
query-scene context. When evidence is supplied, use it only as supporting
context: similarity is not proof and retrieved examples can reinforce a wrong
detector prediction. Do not claim to see information outside the supplied
images. Return only JSON:
{{
  "decision": "accept" | "correct" | "human_review",
  "suggested_class": "one allowed class or null",
  "confidence": 0.0,
  "observations": "short visual observation",
  "uncertainty": "short limitation or ambiguity",
  "evidence_ids": ["E1"]
}}
"""


class NvidiaNimClient:
    def __init__(
        self,
        *,
        model: str,
        cache_root: Path,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key_variable: str = "NIM_API_KEY",
        timeout: float = 120.0,
    ):
        self.model = model
        self.cache_root = Path(cache_root).resolve()
        self.base_url = base_url.rstrip("/")
        self.api_key_variable = api_key_variable
        self.timeout = timeout

    def _api_key(self) -> str:
        value = os.environ.get(self.api_key_variable, "").strip()
        if not value:
            raise RuntimeError(
                f"{self.api_key_variable} is not set. Supply it as an environment "
                "variable before requesting VLM assistance."
            )
        return value

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(canonical.encode()).hexdigest()
        cache_path = self.cache_root / f"{key}.json"
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUS:
                    detail = error.read().decode(errors="replace")[:500]
                    raise RuntimeError(f"NVIDIA NIM {error.code}: {detail}") from error
                last_error = error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            if attempt == 3:
                raise RuntimeError(f"NVIDIA NIM request failed: {last_error}")
            time.sleep(min(2 ** attempt, 30))

        self.cache_root.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def assess(
        self,
        *,
        query_image: Image.Image,
        query_crop: Image.Image,
        detection: DetectionSuggestion,
        classes: Iterable[str],
        mode: str,
        evidence: list[EvidenceRecord] | None = None,
        evidence_images: list[Image.Image] | None = None,
        evidence_crops: list[Image.Image] | None = None,
    ) -> VlmAssessment:
        selected_evidence = list(evidence or [])
        selected_images = list(evidence_images or [])
        selected_crops = list(evidence_crops or [])
        if mode == "query_only":
            selected_evidence = []
            selected_images = []
            selected_crops = []
        elif mode == "retrieval_grounded" and not selected_evidence:
            raise ValueError("Retrieved evidence is required in grounded mode")
        elif mode != "retrieval_grounded":
            raise ValueError(f"Unknown VLM assessment mode: {mode}")

        prompt = build_prompt(detection, selected_evidence, classes, mode)
        montage = grounded_montage(
            query_image,
            query_crop,
            detection,
            selected_evidence,
            selected_images,
            selected_crops,
        )
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image(montage)}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 700,
        }
        payload = self._request(body)
        raw = (payload["choices"][0].get("message") or {}).get("content") or ""
        return parse_assessment(raw, classes)


def parse_assessment(raw: str, classes: Iterable[str]) -> VlmAssessment:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return VlmAssessment(
            decision="human_review",
            suggested_class=None,
            confidence=None,
            observations=raw.strip(),
            uncertainty="The VLM response was not valid structured JSON.",
            evidence_ids=(),
            raw_response=raw,
        )
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return VlmAssessment(
            decision="human_review",
            suggested_class=None,
            confidence=None,
            observations=raw.strip(),
            uncertainty="The VLM response contained invalid JSON.",
            evidence_ids=(),
            raw_response=raw,
        )

    allowed_classes = set(classes)
    suggested = parsed.get("suggested_class")
    if isinstance(suggested, str):
        suggested = suggested.strip().replace(" ", "_")
    if suggested not in allowed_classes:
        suggested = None
    decision = str(parsed.get("decision", "human_review"))
    if decision not in {"accept", "correct", "human_review"}:
        decision = "human_review"
    confidence = parsed.get("confidence")
    try:
        confidence = min(max(float(confidence), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = None
    return VlmAssessment(
        decision=decision,
        suggested_class=suggested,
        confidence=confidence,
        observations=str(parsed.get("observations", "")),
        uncertainty=str(parsed.get("uncertainty", "")),
        evidence_ids=tuple(map(str, parsed.get("evidence_ids") or [])),
        raw_response=raw,
    )
