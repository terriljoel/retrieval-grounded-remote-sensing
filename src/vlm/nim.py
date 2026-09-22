from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
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


@dataclass(frozen=True)
class ImageScreeningAssessment:
    decision: str
    possible_classes: tuple[str, ...]
    confidence: float | None
    observations: str
    uncertainty: str
    suspicious_tiles: tuple[str, ...]
    raw_response: str


def encode_image(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def create_overlapping_tiles(
    image: Image.Image, tile_size: int, overlap_fraction: float
) -> list[tuple[str, Image.Image]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")
    step = max(1, round(tile_size * (1.0 - overlap_fraction)))

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, step))
        final = length - tile_size
        if values[-1] != final:
            values.append(final)
        return values

    tiles = []
    for row, top in enumerate(starts(image.height)):
        for column, left in enumerate(starts(image.width)):
            right = min(left + tile_size, image.width)
            bottom = min(top + tile_size, image.height)
            tiles.append((
                f"T{row + 1}_{column + 1}",
                image.crop((left, top, right, bottom)),
            ))
    return tiles


def build_screening_prompt(
    classes: Iterable[str], tile_ids: Iterable[str]
) -> str:
    readable_classes = ", ".join(name.replace("_", " ") for name in classes)
    readable_tiles = ", ".join(tile_ids)
    return f"""You screen aerial images that received zero object detections.

The first supplied image is the complete scene. The remaining images are
overlapping high-resolution tiles identified in order as: {readable_tiles}.
Allowed target classes: {readable_classes}

Inspect the complete scene for context, then inspect every tile for small target
objects. Use "likely_background" only when no allowed target object is visible
and no tile is suspicious. Use "possible_target_object" when an allowed object
may be present. Use "uncertain" whenever resolution, occlusion, scale, or scene
ambiguity prevents a confident background decision. Prefer human review over
incorrectly clearing a positive image. Do not invent objects.

Return only JSON:
{{
  "decision": "likely_background" | "possible_target_object" | "uncertain",
  "possible_classes": ["allowed_class"],
  "confidence": 0.0,
  "observations": "short visual observation",
  "uncertainty": "short limitation or ambiguity",
  "suspicious_tiles": ["T1_1"]
}}
"""


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
        evidence_ids_example = "[]"
    elif mode == "retrieval_grounded":
        panel_description = (
            "Q full is the complete query image with the proposed box marked and "
            "Q crop is its tight crop. Each E row contains a retrieved verified "
            "example: its complete source image with the verified box marked, then "
            "the corresponding object crop."
        )
        evidence_section = f"Retrieved evidence:\n{evidence_lines}"
        evidence_ids_example = '["E1"]'
    else:
        raise ValueError(f"Unknown VLM assessment mode: {mode}")

    return f"""You assist a human annotating aerial imagery.

{panel_description}

Detector suggestion: {detection.class_name.replace('_', ' ')}
Detector confidence: {detection.confidence:.4f}
Allowed classes: {readable_classes}

{evidence_section}

Follow this decision order strictly:
1. Inspect Q full first. Identify the global scene type and the structures
   surrounding the marked box.
2. Decide whether that global scene is compatible with the proposed class.
3. Inspect Q crop and decide whether its local visual features independently
   support the proposed class.
4. Only after steps 1-3, inspect any retrieved evidence. Use retrieval to
   corroborate the query, never to override contradictory query context.

An "accept" decision is permitted only when BOTH Q full context and Q crop
support the detector class. If the crop resembles the class but the global
scene contradicts it, return "human_review" (or "correct" only when another
allowed class is clearly visible). If retrieval supports the detector but the
query itself does not, return "human_review". A high cosine similarity is not
proof of identity. Retrieved examples can reinforce a wrong prediction.

Do not invent repeated instances in Q full, and do not call surrounding objects
similar examples unless they are visibly so. The correct object may be outside
the allowed class list; in that case return "human_review" with
"suggested_class": null. In "observations", explicitly report three items:
"Global context: ...; Local object: ...; Evidence consistency: ...". State any
conflict between the global context, local crop, detector, and retrieval in
"uncertainty". Do not claim to see information outside the supplied images.
Return only JSON:
{{
  "decision": "accept" | "correct" | "human_review",
  "suggested_class": "one allowed class or null",
  "confidence": 0.0,
  "observations": "short visual observation",
  "uncertainty": "short limitation or ambiguity",
  "evidence_ids": {evidence_ids_example}
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
        requests_per_minute: float = 30.0,
        max_retries: int = 3,
        retry_initial_delay_seconds: float = 1.0,
        retry_delay_increment_seconds: float = 1.0,
    ):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.model = model
        self.cache_root = Path(cache_root).resolve()
        self.base_url = base_url.rstrip("/")
        self.api_key_variable = api_key_variable
        self.timeout = timeout
        self.minimum_request_interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.retry_initial_delay_seconds = retry_initial_delay_seconds
        self.retry_delay_increment_seconds = retry_delay_increment_seconds
        self._last_request_time: float | None = None

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
        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            if self._last_request_time is not None:
                elapsed = time.monotonic() - self._last_request_time
                time.sleep(max(0.0, self.minimum_request_interval - elapsed))
            self._last_request_time = time.monotonic()
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
            if attempt == self.max_retries:
                raise RuntimeError(f"NVIDIA NIM request failed: {last_error}")
            delay = (
                self.retry_initial_delay_seconds
                + attempt * self.retry_delay_increment_seconds
            )
            logging.warning(
                "NVIDIA NIM request attempt %d/%d failed (%s); retrying in "
                "%.1f second(s).",
                attempt + 1,
                total_attempts,
                last_error,
                delay,
            )
            time.sleep(delay)

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

    def screen_image(
        self,
        *,
        image: Image.Image,
        tiles: list[tuple[str, Image.Image]],
        classes: Iterable[str],
    ) -> ImageScreeningAssessment:
        tile_ids = [tile_id for tile_id, _ in tiles]
        prompt = build_screening_prompt(classes, tile_ids)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(image)},
        })
        content.extend({
            "type": "image_url",
            "image_url": {"url": encode_image(tile)},
        } for _, tile in tiles)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 700,
        }
        payload = self._request(body)
        raw = (payload["choices"][0].get("message") or {}).get("content") or ""
        return parse_screening_assessment(raw, classes, tile_ids)


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


def parse_screening_assessment(
    raw: str, classes: Iterable[str], tile_ids: Iterable[str]
) -> ImageScreeningAssessment:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    parsed: dict[str, Any] = {}
    parse_error = ""
    if not match:
        parse_error = "The VLM response was not valid structured JSON."
    else:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            parse_error = "The VLM response contained invalid JSON."

    allowed_classes = set(classes)
    possible_classes = []
    for value in parsed.get("possible_classes") or []:
        normalized = str(value).strip().replace(" ", "_")
        if normalized in allowed_classes and normalized not in possible_classes:
            possible_classes.append(normalized)
    allowed_tiles = set(tile_ids)
    suspicious_tiles = tuple(
        value for value in map(str, parsed.get("suspicious_tiles") or [])
        if value in allowed_tiles
    )
    decision = str(parsed.get("decision", "uncertain"))
    if decision not in {
        "likely_background", "possible_target_object", "uncertain"
    }:
        decision = "uncertain"
    confidence = parsed.get("confidence")
    try:
        confidence = min(max(float(confidence), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = None
    return ImageScreeningAssessment(
        decision=decision,
        possible_classes=tuple(possible_classes),
        confidence=confidence,
        observations=str(parsed.get("observations", raw.strip())),
        uncertainty=parse_error or str(parsed.get("uncertainty", "")),
        suspicious_tiles=suspicious_tiles,
        raw_response=raw,
    )
