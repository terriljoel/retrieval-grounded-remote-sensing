from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from src.annotation.models import Box, DetectionSuggestion
from src.vlm.nim import (
    NvidiaNimClient,
    create_overlapping_tiles,
    parse_screening_assessment,
)


def test_overlapping_tiles_cover_image_edges():
    image = Image.new("RGB", (1000, 700), "gray")
    tiles = create_overlapping_tiles(image, tile_size=512, overlap_fraction=0.2)

    assert len(tiles) == 6
    assert all(tile.width <= 512 and tile.height <= 512 for _, tile in tiles)
    assert len({tile_id for tile_id, _ in tiles}) == len(tiles)


def test_invalid_screening_response_routes_to_uncertain():
    assessment = parse_screening_assessment(
        "not json", ["airplane", "vehicle"], ["T1_1"]
    )

    assert assessment.decision == "uncertain"
    assert assessment.confidence is None
    assert "not valid structured JSON" in assessment.uncertainty


def test_screening_parser_rejects_unknown_classes_and_tiles():
    assessment = parse_screening_assessment(
        '{"decision":"possible_target_object","possible_classes":'
        '["airplane","tree"],"confidence":0.8,"suspicious_tiles":'
        '["T1_1","T9_9"]}',
        ["airplane", "vehicle"],
        ["T1_1"],
    )

    assert assessment.possible_classes == ("airplane",)
    assert assessment.suspicious_tiles == ("T1_1",)


def test_assessment_sends_high_resolution_query_before_montage(tmp_path, monkeypatch):
    captured = {}
    client = NvidiaNimClient(
        model="test-model",
        cache_root=tmp_path,
        query_image_max_size=1024,
    )

    def fake_request(body):
        captured["body"] = body
        return {"choices": [{"message": {"content": (
            '{"decision":"accept","suggested_class":"airplane",'
            '"confidence":0.9,"observations":"test",'
            '"uncertainty":"none","evidence_ids":[]}'
        )}}]}

    monkeypatch.setattr(client, "_request", fake_request)
    query = Image.new("RGB", (1600, 2000), "gray")
    detection = DetectionSuggestion(
        detection_id="det-1",
        class_id=0,
        class_name="airplane",
        confidence=0.9,
        box=Box(100, 100, 300, 300),
    )
    client.assess(
        query_image=query,
        query_crop=query.crop((100, 100, 300, 300)),
        detection=detection,
        classes=("airplane", "ship"),
        mode="query_only",
    )

    content = captured["body"]["messages"][0]["content"]
    assert len(content) == 3
    assert "Stage A -- object validity" in content[0]["text"]
    assert "Retrieved evidence: none" in content[0]["text"]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as high_resolution_query:
        assert high_resolution_query.size == (819, 1024)
    assert content[2]["image_url"]["url"].startswith("data:image/")
