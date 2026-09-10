"""Single-image retrieval-grounded annotation assistant.

Run from the repository root:
    streamlit run app/annotation_assistant.py -- \
        --config configs/annotation/single_image.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.config import load_annotation_config, resolve_runtime_path
from src.annotation.detector import UltralyticsDetector
from src.annotation.models import AnnotationRecord, Box, DetectionSuggestion
from src.annotation.policy import evaluate_acceptance_policy
from src.annotation.service import DetectionEvidence, retrieve_detection_evidence
from src.annotation.storage import save_annotation_session
from src.retrieval.lancedb_store import LanceDbEvidenceStore, find_latest_database
from src.retrieval.remoteclip import RemoteClipEncoder
from src.vlm.nim import NvidiaNimClient, VlmAssessment


BOX_COLOURS = ("#00d4ff", "#ff9f1c", "#2ec4b6", "#e71d36", "#9b5de5")


def parse_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "ANNOTATION_CONFIG", "configs/annotation/single_image.yaml"
        ),
    )
    arguments, _ = parser.parse_known_args()
    path = Path(arguments.config)
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_resource(show_spinner="Loading detector...")
def load_detector(checkpoint: str) -> UltralyticsDetector:
    return UltralyticsDetector(Path(checkpoint))


@st.cache_resource(show_spinner="Loading RemoteCLIP...")
def load_encoder(
    model_name: str,
    repository: str,
    filename: str,
    cache_dir: str,
    device: str | None,
    dimension: int,
) -> RemoteClipEncoder:
    return RemoteClipEncoder(
        model_name=model_name,
        repository=repository,
        filename=filename,
        cache_dir=Path(cache_dir),
        device=device,
        expected_dimension=dimension,
    )


@st.cache_resource(show_spinner="Opening LanceDB...")
def load_evidence_store(
    database_path: str,
    table_name: str,
    dataset_root: str,
) -> LanceDbEvidenceStore:
    return LanceDbEvidenceStore(
        database_path=Path(database_path),
        table_name=table_name,
        dataset_root=Path(dataset_root),
    )


def detector_models(config: dict) -> dict[str, Path]:
    detector = config["detector"]
    configured = detector.get("models")
    if configured:
        return {
            name: resolve_runtime_path(path, PROJECT_ROOT)
            for name, path in configured.items()
        }
    return {
        detector.get("name", "detector"): resolve_runtime_path(
            detector["checkpoint"], PROJECT_ROOT
        )
    }


def resolve_database_path(config: dict) -> Path:
    retrieval = config["retrieval"]
    if retrieval.get("database_path"):
        return resolve_runtime_path(retrieval["database_path"], PROJECT_ROOT)
    artifact_root = resolve_runtime_path(retrieval["artifact_root"], PROJECT_ROOT)
    if retrieval.get("artifact", "latest") != "latest":
        return (artifact_root / retrieval["artifact"] / "lancedb").resolve()
    return find_latest_database(artifact_root)


def reset_image_state(image_key: str) -> None:
    if st.session_state.get("image_key") == image_key:
        return
    for key in (
        "detections",
        "annotations",
        "rejected_detection_ids",
        "evidence",
        "evidence_key",
        "vlm_assessments",
        "vlm_key",
        "review_records",
        "saved_directory",
    ):
        st.session_state.pop(key, None)
    st.session_state["image_key"] = image_key
    st.session_state["annotations"] = []
    st.session_state["rejected_detection_ids"] = []
    st.session_state["review_records"] = {}


def draw_boxes(
    image: Image.Image,
    detections: list[DetectionSuggestion],
    selected_id: str | None,
) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    for index, detection in enumerate(detections):
        selected = detection.detection_id == selected_id
        colour = BOX_COLOURS[index % len(BOX_COLOURS)]
        width = 4 if selected else 2
        draw.rectangle(detection.box.as_list(), outline=colour, width=width)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = draw.textbbox((detection.box.xmin, detection.box.ymin), label)
        draw.rectangle(text_box, fill=colour)
        draw.text((detection.box.xmin, detection.box.ymin), label, fill="black")
    return rendered


def edited_detection(
    detection: DetectionSuggestion,
    classes: tuple[str, ...],
    image: Image.Image,
) -> DetectionSuggestion:
    class_name = st.selectbox(
        "Annotation class",
        classes,
        index=classes.index(detection.class_name),
        key=f"class_{detection.detection_id}",
    )
    columns = st.columns(4)
    values = []
    specifications = (
        ("xmin", detection.box.xmin, 0.0, float(image.width - 1)),
        ("ymin", detection.box.ymin, 0.0, float(image.height - 1)),
        ("xmax", detection.box.xmax, 1.0, float(image.width)),
        ("ymax", detection.box.ymax, 1.0, float(image.height)),
    )
    for column, (name, value, minimum, maximum) in zip(
        columns, specifications, strict=True
    ):
        with column:
            values.append(st.number_input(
                name,
                min_value=minimum,
                max_value=maximum,
                value=float(value),
                step=1.0,
                key=f"{name}_{detection.detection_id}",
            ))
    box = Box(*values)
    box.validate(image.width, image.height)
    return replace(
        detection,
        class_id=classes.index(class_name),
        class_name=class_name,
        box=box,
    )


def evidence_identity(detection: DetectionSuggestion) -> tuple:
    return (
        detection.detection_id,
        detection.class_id,
        *detection.box.as_list(),
    )


def render_evidence(
    title: str,
    records,
    store: LanceDbEvidenceStore,
) -> list[Image.Image]:
    st.subheader(title)
    if not records:
        st.info("No eligible evidence found.")
        return []
    columns = st.columns(len(records))
    crops = []
    for rank, (column, record) in enumerate(zip(columns, records, strict=True), start=1):
        crop = store.evidence_crop(record)
        crops.append(crop)
        with column:
            st.image(crop, use_container_width=True)
            st.caption(
                f"E{rank} · {record.class_name}\n"
                f"cosine {record.cosine_similarity:.3f} · {record.split}"
            )
    return crops


def upsert_annotation(record: AnnotationRecord) -> None:
    existing = [
        annotation
        for annotation in st.session_state["annotations"]
        if annotation.source_detection_id != record.source_detection_id
    ]
    existing.append(record)
    st.session_state["annotations"] = existing
    rejected = set(st.session_state["rejected_detection_ids"])
    rejected.discard(record.source_detection_id)
    st.session_state["rejected_detection_ids"] = sorted(rejected)


def evidence_summary(record) -> dict:
    return {
        "embedding_id": record.embedding_id,
        "image_id": record.image_id,
        "image_key": record.image_key,
        "split": record.split,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "crop_type": record.crop_type,
        "cosine_similarity": record.cosine_similarity,
    }


def assessment_summary(model: str, assessment: VlmAssessment) -> dict:
    return {
        "model": model,
        "decision": assessment.decision,
        "suggested_class": assessment.suggested_class,
        "confidence": assessment.confidence,
        "observations": assessment.observations,
        "uncertainty": assessment.uncertainty,
        "evidence_ids": list(assessment.evidence_ids),
        "raw_response": assessment.raw_response,
    }


def render_assessment(title: str, assessment: VlmAssessment) -> None:
    st.subheader(title)
    st.write(f"**Recommendation:** {assessment.decision}")
    st.write(f"**Suggested class:** {assessment.suggested_class or 'none'}")
    if assessment.confidence is not None:
        st.write(f"**Reported confidence:** {assessment.confidence:.3f}")
    st.write(f"**Observation:** {assessment.observations}")
    st.write(f"**Uncertainty:** {assessment.uncertainty}")


def single_image_mode(config: dict) -> None:
    classes = tuple(config["classes"])
    models = detector_models(config)
    default_name = config["detector"].get("default_model")
    model_names = list(models)
    default_index = model_names.index(default_name) if default_name in models else 0
    model_name = st.sidebar.selectbox("Detector", model_names, index=default_index)
    with st.sidebar.expander("Runtime status"):
        checkpoint = models[model_name]
        st.write(f"Checkpoint: {'ready' if checkpoint.is_file() else 'missing'}")
        st.caption(str(checkpoint))
        try:
            database_path = resolve_database_path(config)
            st.write("LanceDB: ready")
            st.caption(str(database_path))
        except (FileNotFoundError, ValueError) as error:
            st.write("LanceDB: missing")
            st.caption(str(error))
        key_variable = config["vlm"]["api_key_variable"]
        st.write(
            f"VLM key: {'set' if os.environ.get(key_variable, '').strip() else 'not set'}"
        )
    uploaded = st.file_uploader(
        "Upload one remote-sensing image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )
    if uploaded is None:
        st.info("Upload one image to begin annotation.")
        return

    image_bytes = uploaded.getvalue()
    image_key = hashlib.sha256(image_bytes).hexdigest()
    reset_image_state(image_key)
    image = Image.open(uploaded).convert("RGB")

    if st.button("Run detector", type="primary"):
        detector = load_detector(str(models[model_name]))
        settings = config["detector"]
        with st.spinner("Running object detection..."):
            st.session_state["detections"] = detector.predict(
                image,
                confidence=float(settings["confidence"]),
                iou=float(settings["iou"]),
                image_size=int(settings["image_size"]),
                device=settings.get("device"),
            )

    detections: list[DetectionSuggestion] | None = st.session_state.get("detections")
    if detections is None:
        st.image(image, caption=uploaded.name, use_container_width=True)
        return
    if not detections:
        st.warning("The detector found no objects. Use the manual annotation section.")
        selected = None
    else:
        labels = [
            f"{item.detection_id}: {item.class_name} ({item.confidence:.3f})"
            for item in detections
        ]
        selected_index = st.selectbox(
            "Detection to review", range(len(detections)), format_func=labels.__getitem__
        )
        selected = detections[selected_index]
        st.image(
            draw_boxes(image, detections, selected.detection_id),
            caption=f"{len(detections)} detector suggestions",
            use_container_width=True,
        )

    if selected is not None:
        st.header("Review selected detection")
        try:
            review = edited_detection(selected, classes, image)
        except ValueError as error:
            st.error(str(error))
            return

        button_columns = st.columns(3)
        with button_columns[0]:
            if st.button("Accept / update"):
                unchanged = (
                    review.class_id == selected.class_id and review.box == selected.box
                )
                upsert_annotation(AnnotationRecord(
                    annotation_id=f"ann_{selected.detection_id}",
                    class_id=review.class_id,
                    class_name=review.class_name,
                    box=review.box,
                    decision="accepted" if unchanged else "corrected",
                    source_detection_id=selected.detection_id,
                    detector_confidence=selected.confidence,
                ))
                reviews = dict(st.session_state["review_records"])
                review_record = dict(reviews.get(selected.detection_id, {}))
                review_record["human_decision"] = (
                    "accepted" if unchanged else "corrected"
                )
                review_record["final_class"] = review.class_name
                review_record["final_box"] = review.box.as_list()
                reviews[selected.detection_id] = review_record
                st.session_state["review_records"] = reviews
                st.success("Annotation kept in this session.")
        with button_columns[1]:
            if st.button("Reject detection"):
                st.session_state["annotations"] = [
                    item for item in st.session_state["annotations"]
                    if item.source_detection_id != selected.detection_id
                ]
                rejected = set(st.session_state["rejected_detection_ids"])
                rejected.add(selected.detection_id)
                st.session_state["rejected_detection_ids"] = sorted(rejected)
                reviews = dict(st.session_state["review_records"])
                review_record = dict(reviews.get(selected.detection_id, {}))
                review_record["human_decision"] = "rejected"
                reviews[selected.detection_id] = review_record
                st.session_state["review_records"] = reviews
                st.warning("Detection marked as rejected.")

        database_path = resolve_database_path(config)
        retrieval = config["retrieval"]
        store = load_evidence_store(
            str(database_path),
            retrieval["table"],
            str(resolve_runtime_path(retrieval["dataset_root"], PROJECT_ROOT)),
        )
        with button_columns[2]:
            if st.button("Retrieve evidence"):
                embedding = config["embedding"]
                encoder = load_encoder(
                    embedding["model_name"],
                    embedding["repository"],
                    embedding["filename"],
                    str(resolve_runtime_path(embedding["cache_dir"], PROJECT_ROOT)),
                    embedding.get("device"),
                    int(embedding["dimension"]),
                )
                with st.spinner("Embedding the query and searching LanceDB..."):
                    st.session_state["evidence"] = retrieve_detection_evidence(
                        image=image,
                        detection=review,
                        encoder=encoder,
                        store=store,
                        context_margin=float(embedding["context_margin"]),
                        evidence_splits=tuple(retrieval["evidence_splits"]),
                        top_k=int(retrieval["top_k"]),
                        candidate_limit=int(retrieval["candidate_limit"]),
                    )
                    st.session_state["evidence_key"] = evidence_identity(review)
                    st.session_state.pop("vlm_assessments", None)
                    retrieved = st.session_state["evidence"]
                    reviews = dict(st.session_state["review_records"])
                    review_record = dict(reviews.get(selected.detection_id, {}))
                    review_record.update({
                        "detector_class": selected.class_name,
                        "detector_confidence": selected.confidence,
                        "reviewed_class": review.class_name,
                        "reviewed_box": review.box.as_list(),
                        "object_evidence": [
                            evidence_summary(item) for item in retrieved.object_cases
                        ],
                        "context_evidence": [
                            evidence_summary(item) for item in retrieved.context_cases
                        ],
                    })
                    reviews[selected.detection_id] = review_record
                    st.session_state["review_records"] = reviews

        evidence: DetectionEvidence | None = st.session_state.get("evidence")
        evidence_is_current = (
            evidence is not None
            and st.session_state.get("evidence_key") == evidence_identity(review)
        )
        if evidence is not None and not evidence_is_current:
            st.warning("The box or class changed. Retrieve evidence again.")
        if evidence_is_current:
            tabs = st.tabs(["Object evidence", "Context evidence"])
            with tabs[0]:
                object_crops = render_evidence(
                    "Object-only neighbours", evidence.object_cases, store
                )
            with tabs[1]:
                render_evidence("Context neighbours", evidence.context_cases, store)

            vlm_config = config["vlm"]
            if vlm_config.get("enabled", True):
                mode_labels = {
                    "Query only": "query_only",
                    "Query + retrieved evidence": "retrieval_grounded",
                    "Compare both": "compare",
                }
                configured_mode = vlm_config.get(
                    "default_mode", "retrieval_grounded"
                )
                default_label = next(
                    label for label, value in mode_labels.items()
                    if value == configured_mode
                )
                selected_label = st.radio(
                    "VLM input",
                    tuple(mode_labels),
                    index=tuple(mode_labels).index(default_label),
                    horizontal=True,
                    key=f"vlm_mode_{review.detection_id}",
                )
                selected_mode = mode_labels[selected_label]
                max_evidence = int(
                    vlm_config.get("max_evidence", len(evidence.object_cases))
                )
                vlm_evidence = evidence.object_cases[:max_evidence]
                vlm_crops = object_crops[:max_evidence]
                if st.button("Run VLM assessment"):
                    client = NvidiaNimClient(
                        model=vlm_config["model"],
                        cache_root=resolve_runtime_path(
                            vlm_config["cache_root"], PROJECT_ROOT
                        ),
                        base_url=vlm_config["base_url"],
                        api_key_variable=vlm_config["api_key_variable"],
                        timeout=float(vlm_config["timeout"]),
                    )
                    requested_modes = (
                        ("query_only", "retrieval_grounded")
                        if selected_mode == "compare"
                        else (selected_mode,)
                    )
                    with st.spinner("Requesting VLM assessment..."):
                        evidence_images = [
                            store.evidence_image(record) for record in vlm_evidence
                        ]
                        assessments = {}
                        for mode in requested_modes:
                            assessments[mode] = client.assess(
                                query_image=image,
                                query_crop=evidence.object_crop,
                                detection=review,
                                classes=classes,
                                mode=mode,
                                evidence=vlm_evidence,
                                evidence_images=evidence_images,
                                evidence_crops=vlm_crops,
                            )
                        st.session_state["vlm_assessments"] = assessments
                        st.session_state["vlm_key"] = evidence_identity(review)
                        reviews = dict(st.session_state["review_records"])
                        review_record = dict(reviews.get(selected.detection_id, {}))
                        review_record["vlm"] = {
                            mode: assessment_summary(vlm_config["model"], assessment)
                            for mode, assessment in assessments.items()
                        }
                        grounded = assessments.get("retrieval_grounded")
                        if grounded is not None:
                            policy_result = evaluate_acceptance_policy(
                                detection=review,
                                evidence=vlm_evidence,
                                assessment=grounded,
                                settings=config["decision_policy"],
                            )
                            review_record["decision_policy"] = policy_result.to_dict()
                        reviews[selected.detection_id] = review_record
                        st.session_state["review_records"] = reviews

                assessments: dict[str, VlmAssessment] | None = st.session_state.get(
                    "vlm_assessments"
                )
                if (
                    assessments
                    and st.session_state.get("vlm_key") == evidence_identity(review)
                ):
                    columns = st.columns(len(assessments))
                    for column, (mode, assessment) in zip(
                        columns, assessments.items(), strict=True
                    ):
                        with column:
                            title = (
                                "Query only"
                                if mode == "query_only"
                                else "Retrieval-grounded"
                            )
                            render_assessment(title, assessment)
                            if mode == "retrieval_grounded":
                                policy_result = evaluate_acceptance_policy(
                                    detection=review,
                                    evidence=vlm_evidence,
                                    assessment=assessment,
                                    settings=config["decision_policy"],
                                )
                                st.write(
                                    "**Configured policy recommendation:** "
                                    f"{policy_result.recommendation}"
                                )
                                for reason in policy_result.reasons:
                                    st.caption(f"Gate not passed: {reason}")
                    st.caption(
                        "Both variants see the boxed full query image and tight crop. "
                        "Only the grounded variant sees boxed full evidence images and "
                        "their crops. The human annotator still makes the final decision."
                    )

    with st.expander("Add a missed object manually"):
        manual_class = st.selectbox("Manual class", classes, key="manual_class")
        columns = st.columns(4)
        defaults = (0.0, 0.0, float(image.width), float(image.height))
        limits = (
            (0.0, float(image.width - 1)),
            (0.0, float(image.height - 1)),
            (1.0, float(image.width)),
            (1.0, float(image.height)),
        )
        manual_values = []
        for column, name, value, limit in zip(
            columns, ("xmin", "ymin", "xmax", "ymax"), defaults, limits, strict=True
        ):
            with column:
                manual_values.append(st.number_input(
                    name,
                    min_value=limit[0],
                    max_value=limit[1],
                    value=value,
                    step=1.0,
                    key=f"manual_{name}",
                ))
        if st.button("Add manual annotation"):
            try:
                manual_box = Box(*manual_values)
                manual_box.validate(image.width, image.height)
                index = sum(
                    item.source_detection_id is None
                    for item in st.session_state["annotations"]
                )
                st.session_state["annotations"].append(AnnotationRecord(
                    annotation_id=f"manual_{index:04d}",
                    class_id=classes.index(manual_class),
                    class_name=manual_class,
                    box=manual_box,
                    decision="manual",
                    source_detection_id=None,
                    detector_confidence=None,
                ))
                st.success("Manual annotation added to this session.")
            except ValueError as error:
                st.error(str(error))

    st.header("Annotation session")
    annotations: list[AnnotationRecord] = st.session_state["annotations"]
    st.write(f"Kept annotations: {len(annotations)}")
    st.write(f"Rejected suggestions: {len(st.session_state['rejected_detection_ids'])}")
    if annotations:
        st.dataframe([record.to_dict() for record in annotations], use_container_width=True)
    if st.button("Save verified annotation", disabled=not annotations):
        output_root = resolve_runtime_path(
            config["annotations"]["output_root"], PROJECT_ROOT
        )
        run_directory = save_annotation_session(
            image=image,
            original_filename=uploaded.name,
            annotations=annotations,
            output_root=output_root,
            metadata={
                "detector_name": model_name,
                "detector_checkpoint": str(models[model_name]),
                "lancedb_path": str(resolve_database_path(config)),
                "rejected_detection_ids": st.session_state["rejected_detection_ids"],
                "reviews": st.session_state["review_records"],
            },
        )
        st.session_state["saved_directory"] = str(run_directory)
    if st.session_state.get("saved_directory"):
        st.success(f"Saved to {st.session_state['saved_directory']}")


def main() -> None:
    st.set_page_config(page_title="Remote-Sensing Annotation Assistant", layout="wide")
    local_shared_resources = PROJECT_ROOT / "shared_resources"
    if "SHARED_RESOURCES_ROOT" not in os.environ and local_shared_resources.is_dir():
        os.environ["SHARED_RESOURCES_ROOT"] = str(local_shared_resources.resolve())
    try:
        config = load_annotation_config(parse_config_path())
    except Exception as error:
        st.error(f"Configuration error: {error}")
        st.stop()
        return

    st.title(config["app"]["title"])
    mode = st.sidebar.radio("Annotation mode", ("Single image", "Batch"))
    if mode == "Batch":
        st.header("Batch annotation")
        st.info(
            "Batch mode is intentionally deferred. It will reuse the validated "
            "single-image detection, retrieval, VLM, and save services."
        )
        return
    single_image_mode(config)


if __name__ == "__main__":
    main()
