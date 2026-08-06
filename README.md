# Retrieval-Grounded Remote Sensing

A deep-learning project for detecting and verifying objects in very-high-resolution remote-sensing imagery.

The project starts with a YOLOv8 object-detection baseline using the NWPU VHR-10 dataset. Later stages will use image embeddings, FAISS semantic retrieval, and a vision-language model (VLM) to verify detector predictions.

## Project objectives

1. Train and evaluate an object detector on NWPU VHR-10.
2. Analyse false positives, false negatives, and localisation errors.
3. Generate embeddings for detected objects and reference images.
4. Retrieve similar examples using FAISS.
5. Use a VLM and retrieved evidence to recommend accept, reject, relabel, or human review.
6. Compare detector-only, same-class retrieval, and semantic-retrieval systems.

## Dataset

The initial experiments use NWPU VHR-10 with ten classes: airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, and vehicle. The original dataset is stored outside Git and is not committed to GitHub.

## Repository structure

```text
retrieval-grounded-remote-sensing/
|-- configs/                 # Dataset and experiment configurations
|-- docs/                    # Decisions and project documentation
|-- notebooks/               # Jupyter and Colab notebooks
|-- shared_resources/        # Small, versionable shared files
|-- src/                     # Reusable Python modules
|-- scripts/                 # Command-line entry points
|-- tests/                   # Automated tests
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Automated object-detection baseline

The validated notebook workflow is available as configuration-driven CLI commands. The current backend is Ultralytics, while model weights are a YAML setting, so YOLO nano, small, and medium variants do not require code changes.

For Colab, open `notebooks/00_colab_bootstrap.ipynb` directly from GitHub and run its setup cell once. It mounts Drive, clones or updates this repository under `/content`, installs the project, and configures the runtime paths. The bootstrap currently selects the `feat/object-detection-YOLO` branch; change `REPO_REF` to `main` after merging. The training notebook remains independent and is not modified by the bootstrap.

Install the repository:

```bash
python -m pip install -e .
```

For local development and tests, use `python -m pip install -e ".[dev]"`.

Set runtime paths. In Colab, raw and processed datasets should normally use fast temporary `/content` storage, while manifests and experiment results should use persistent Google Drive storage:

```bash
export SHARED_RESOURCES_ROOT="/content/drive/Othercomputers/My laptop/shared_resources"
export RAW_DATASET_ROOT="/content/datasets/raw"
export PROCESSED_DATASET_ROOT="/content/datasets/processed"
export MANIFEST_ROOT="$SHARED_RESOURCES_ROOT/datasets/manifests"
export EXPERIMENT_OUTPUT_ROOT="$SHARED_RESOURCES_ROOT/experiment_outputs"
```

Prepare or reuse the fixed split and validated YOLO export:

```bash
rs-prepare-detector --config configs/detection/yolov8n_baseline.yaml
```

Train and select `best.pt` using validation data:

```bash
rs-train-detector --config configs/detection/yolov8n_baseline.yaml
```

Evaluate the selected checkpoint once on the held-out test split:

```bash
rs-evaluate-detector \
  --config configs/detection/yolov8n_baseline.yaml \
  --checkpoint /path/to/best.pt \
  --confirm-test
```

Export structured predictions independently from training. Copy the example
inference YAML, set its checkpoint and source paths, and run the job with that
configuration alone:

```bash
rs-export-predictions --config configs/inference/ultralytics_export.yaml
```

This writes `inference_images.csv`, including images with zero detections, and
`detections.csv`, containing one row per predicted box. The `source.manifest`
setting is optional and only enriches matching images with train/val/test
provenance. A new dataset does not need to be split or converted to YOLO merely
to run inference.

Ground-truth comparison is also optional and independent of the manifest. Add
a `ground_truth` mapping to the inference YAML to compare against YOLO labels:

```yaml
ground_truth:
  format: yolo
  path: /path/to/labels
  iou_threshold: 0.50
```

Use `format: nwpu` for original NWPU annotation text files. Labels are matched
to images by relative path and then filename stem. Missing label files are
treated as background images. The export writes `ground_truth.csv` and
`prediction_comparison.csv` with `true_positive`, `class_error`,
`false_positive`, and `false_negative` records. Set `ground_truth: null` for
prediction-only jobs.

For the prepared NWPU validation set, configure `source.path` as its
`images/val` directory and `source.split: val`. Test export remains protected:
set `source.allow_test: true` explicitly after freezing the inference
configuration. Test predictions are queries only and must not be added to the
FAISS reference index.

Predict arbitrary unlabeled images without annotation conversion:

```bash
rs-predict-detector \
  --config configs/detection/yolov8n_baseline.yaml \
  --checkpoint /path/to/best.pt \
  --source /path/to/images
```

The preparation command reuses a saved audit when available, reuses or creates the deterministic manifest, exports collision-safe YOLO filenames, records source provenance, and validates image, label, object, background, and coordinate contracts. Pass `--force` only when you deliberately want to rerun the audit and replace the manifest and export. Training does not launch test evaluation, and test evaluation requires the explicit confirmation flag.

## Job logs

Every `rs-*` CLI invocation is treated as a job. When `JOB_LOG_ROOT` is set,
the complete console output is shown live and also persisted under:

```text
${JOB_LOG_ROOT}/<command>/<timestamp_job-id>/
|-- run.log
`-- job.json
```

`job.json` records the command, arguments, start and finish times, duration,
status, exit code, working directory, Python version, and log path. The Colab
bootstrap sets `JOB_LOG_ROOT` to
`${EXPERIMENT_OUTPUT_ROOT}/job_logs`, which is stored in the shared Drive
folder. Training, evaluation, and inference metadata also record the active
job ID and job directory.
