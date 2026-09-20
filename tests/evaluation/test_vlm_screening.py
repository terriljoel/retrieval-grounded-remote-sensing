from __future__ import annotations

from PIL import Image

from src.vlm.nim import create_overlapping_tiles, parse_screening_assessment


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
