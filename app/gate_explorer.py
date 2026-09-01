"""Gate explorer -- a six-step demo of the retrieval + VLM gate on real detector proposals.

    image dropdown -> tile with boxes -> detection dropdown ->
    "Find similar cases" -> "Ask the VLM" -> gate verdict and ground truth

One file, no sidebar, no filters. Nothing is embedded, retrieved, or sent to
the VLM until the corresponding button is pressed -- picking an image or a
detection only reads CSVs and draws boxes on the tile.

Run with:
    streamlit run app/gate_explorer.py

The project root is resolved from this file's own path, so it runs without
PYTHONPATH set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.models import BoundingBox
from src.retrieval.crops import write_streams
from src.retrieval.embed import embed_stream
from src.retrieval.export_join import load_split
from src.retrieval.fuse import TOP_K, evidence, fused_similarity
from src.retrieval.memory import decision_mask, load_memory, stream_matrix
from src.retrieval.nim import encode_image, model_name
from src.retrieval.phase1 import DATASET_ROOT, ENCODER, MAIN_SCALE, selected_alpha
from src.retrieval.sensor import sensor_table
from scripts.run_b0 import (
    DEFAULT_DETECTIONS, DEFAULT_SOURCE_DIR, build_real_proposals, image_paths, load_detections,
    load_ground_truth,
)
from scripts.run_b3_vlm import build_prompt, evidence_block, score_logprob

SELECTED = "#00d4ff"
VERDICT_COLOUR = {
    "accept": "#2ecc71",
    "adjust": "#f1c40f",
    "relabel": "#ff1493",
    "reject_localisation": "#ff1493",
    "reject_background": "#ff1493",
}
REGIONAL_STREAM = f"regional_k{MAIN_SCALE:g}"
# Recalibrated 2026-08-31 on REAL detector proposals (see docs/STATE.md,
# "Threshold recalibration") -- the old 0.9436 came from Phase 0's clean-GT
# background distribution and was recalibrated on real accept vs real
# reject_background sim@1 (n=486 / n=189) at this project's standard 1%
# false-accept budget (gate.rejection_threshold, background_reject_rate=0.99).
# The hypothesis going in was that real crops sit lower and a recalibrated
# threshold would be more permissive -- it wasn't: the real-data threshold
# came out HIGHER (0.9601, not lower) at every operating point tested
# (90/95/99%), because real `reject_background` includes NWPU annotation-gap
# detections (real objects the GT never boxed) that score almost as high as
# real accepts, inflating the background distribution's upper tail. This
# threshold escalates MORE real accepts than the old one, not fewer -- that
# is the honest number, not a fix. The 5%-leak alternative, matching Phase
# 0's own construction exactly, is 0.9487 (also higher than the old value).
NOVELTY_ACCEPT_THRESHOLD = 0.9601


@st.cache_resource
def load_context():
    """Everything needed to populate the dropdowns: CSV joins and the case memory.

    No image is opened and nothing is embedded here -- that is what makes the
    later steps the ones that actually "do" something.
    """
    images_csv = DEFAULT_DETECTIONS.with_name("inference_images.csv")
    if not images_csv.is_file():
        images_csv = None
    source = ROOT / DATASET_ROOT / DEFAULT_SOURCE_DIR
    detections = load_detections(DEFAULT_DETECTIONS, images_csv, source)
    split = load_split(ROOT)
    gt = load_ground_truth(ROOT)
    proposals = build_real_proposals(ROOT, detections, split, gt)
    if not proposals:
        raise ValueError("No detections landed on a val-split image.")

    by_image: dict[str, list[dict]] = {}
    for record in proposals:
        by_image.setdefault(record["image_uid"], []).append(record)
    for records in by_image.values():
        records.sort(key=lambda r: r["proposal_id"])

    rows, embeddings, streams = load_memory(
        ROOT / "data" / "memory" / "cases.db", ROOT / "data" / "memory" / "embeddings.npy"
    )
    alpha = selected_alpha(ROOT / "reports" / "phase1" / "alpha_sweep_summary.csv")
    return {
        "by_image": by_image,
        "paths": image_paths(ROOT),
        "rows": rows,
        "positive_mask": decision_mask(rows),
        "memory_local": stream_matrix(embeddings, streams, "local"),
        "memory_regional": stream_matrix(embeddings, streams, REGIONAL_STREAM),
        "alpha": alpha,
    }


def tile_with_boxes(image_path: Path, records: list[dict], selected_id: str) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for record in records:
        selected = record["proposal_id"] == selected_id
        draw.rectangle(
            [record["xmin"], record["ymin"], record["xmax"], record["ymax"]],
            outline=SELECTED if selected else "#888888", width=3 if selected else 1,
        )
    return image


def ensure_crop(proposal: dict) -> None:
    """Write the three crop streams for one proposal, if not already cached."""
    box = BoundingBox(proposal["xmin"], proposal["ymin"], proposal["xmax"], proposal["ymax"], 0)
    with Image.open(load_context()["paths"][proposal["image_uid"]]) as source:
        write_streams(
            source.convert("RGB"), box, proposal["crop_id"],
            ROOT / "data" / "crops" / proposal["corpus"], regional_scales=(MAIN_SCALE,),
        )


def embed_proposal(proposal: dict, stream: str) -> np.ndarray:
    """One crop's embedding. `force=True`: a size-1 batch always maps to the
    same cache filename (embedding_path keys on row count, not content), so
    caching it would silently serve a different proposal's stale vector."""
    return embed_stream(
        [proposal["crop_id"]], ROOT / "data" / "crops" / proposal["corpus"],
        ROOT / "data" / "embeddings" / proposal["corpus"], ENCODER, stream, force=True,
    )[0]


def crop_path_for_case(case: dict) -> Path:
    corpus, crop_id = case["proposal_id"].split(":", 1)
    return ROOT / "data" / "crops" / corpus / "local" / f"{crop_id}.jpg"


@st.cache_data
def similar_cases(proposal_id: str) -> dict:
    context = load_context()
    proposal = next(r for image in context["by_image"].values() for r in image
                     if r["proposal_id"] == proposal_id)
    ensure_crop(proposal)

    query_local = embed_proposal(proposal, "local")[None, :]
    query_regional = embed_proposal(proposal, REGIONAL_STREAM)[None, :]
    similarity = fused_similarity(
        query_local, query_regional,
        context["memory_local"], context["memory_regional"], context["alpha"],
    )
    result = evidence(similarity, context["positive_mask"], TOP_K)

    cases = []
    for rank in range(TOP_K):
        case = context["rows"][int(result["positive_idx"][0, rank])]
        cases.append({
            "label": case["verified_class"],
            "similarity": float(result["positive_sim"][0, rank]),
            "crop_path": crop_path_for_case(case),
            "agrees": case["verified_class"] == proposal["predicted_class"],
        })
    similarities = [c["similarity"] for c in cases]
    return {
        "cases": cases, "top1": similarities[0], "median5": float(np.median(similarities)),
        "agree": sum(c["agrees"] for c in cases),
    }


@st.cache_data
def vlm_verdict(proposal_id: str) -> dict:
    context = load_context()
    proposal = next(r for image in context["by_image"].values() for r in image
                     if r["proposal_id"] == proposal_id)
    similar = similar_cases(proposal_id)
    evidence_text = evidence_block(
        [c["label"] for c in similar["cases"]], [c["similarity"] for c in similar["cases"]],
    )
    prompt = build_prompt(proposal["predicted_class"], evidence_text)
    image_url = encode_image(Image.open(
        ROOT / "data" / "crops" / proposal["corpus"] / "local" / f"{proposal['crop_id']}.jpg"
    ))
    p_yes = score_logprob(
        image_url, proposal["predicted_class"], model_name(),
        ROOT / "data" / "nim_cache", evidence_text=evidence_text,
    )
    return {"prompt": prompt, "p_yes": p_yes}


def main() -> None:
    st.set_page_config(page_title="Gate explorer", layout="centered")
    st.title("Gate explorer")

    try:
        context = load_context()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
        return

    image_uid = st.selectbox("Image", sorted(context["by_image"]))
    records = context["by_image"][image_uid]

    detection_labels = [
        f"{record['crop_id'].rsplit('_det', 1)[-1]}: {record['predicted_class']} "
        f"({record['detector_confidence']:.2f})"
        for record in records
    ]
    choice = st.selectbox("Detection", range(len(records)), format_func=lambda i: detection_labels[i])
    proposal = records[choice]

    st.image(tile_with_boxes(context["paths"][image_uid], records, proposal["proposal_id"]))

    if st.button("Find similar cases"):
        st.session_state["similar"] = similar_cases(proposal["proposal_id"])

    similar = st.session_state.get("similar")
    if similar is not None:
        columns = st.columns(TOP_K)
        for column, case in zip(columns, similar["cases"]):
            with column:
                if case["crop_path"].is_file():
                    st.image(str(case["crop_path"]))
                st.caption(f"{case['label']}\n{case['similarity']:.3f}")
        st.write(
            f"top-1 {similar['top1']:.3f} · median of top-5 {similar['median5']:.3f} · "
            f"{similar['agree']}/{TOP_K} agree with {proposal['predicted_class']}"
        )

    if st.button("Ask the VLM"):
        st.session_state["vlm"] = vlm_verdict(proposal["proposal_id"])

    vlm = st.session_state.get("vlm")
    if vlm is not None:
        st.write(f"p(yes, is a {proposal['predicted_class']}) = {vlm['p_yes']:.3f}")
        with st.expander("Prompt sent"):
            st.text(vlm["prompt"])

    st.divider()
    if similar is not None:
        gate = "accept" if similar["top1"] >= NOVELTY_ACCEPT_THRESHOLD else "escalate"
        st.write(f"**Gate verdict** (sim@1 vs {NOVELTY_ACCEPT_THRESHOLD:.3f}): {gate}")
    else:
        st.write("**Gate verdict:** run \"Find similar cases\" first")

    colour = VERDICT_COLOUR.get(proposal["verdict"], "#888888")
    st.markdown(
        f"**Ground truth:** <span style='color:{colour}'>{proposal['verdict']}</span> "
        f"&nbsp;&nbsp; {proposal['verified_class'] or 'background'} "
        f"(IoU {proposal['iou']:.2f})",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
