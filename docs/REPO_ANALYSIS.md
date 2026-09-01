# REPO ANALYSIS

Analysis of the repository as it exists at commit `85f059c` (branch `feat/object-detection-YOLO`),
read against `docs/PROJECT_BRIEF.md`. Source was read; filenames and docstrings were not trusted.
Where a notebook output and a config disagree, both are reported.

**One-line summary.** The repo is a clean, well-tested *detector-and-export* pipeline for NWPU VHR-10
and nothing else. Steps 1–7 of the work order do not exist. Step 0 is roughly 60% there but rests on
a split artifact that does not match the trained checkpoint, is not committed to Git, and is not
loadable on Linux.

---

## 1. Inventory

Every module, actual responsibility, real I/O, real callers. "Callers" means code that imports it,
not code that could.

### `src/data/`

| Module | Actual responsibility | Real inputs → outputs | Callers |
|---|---|---|---|
| `models.py` (17 L) | `BoundingBox(xmin,ymin,xmax,ymax,class_id)` and `DatasetItem(image_path, annotation_path, is_background)`. Frozen dataclasses, absolute-pixel ints. | — | `audit.py`, `yolo.py`, `splitting.py`, `manifests.py`, `parsers/nwpu.py`, `adapters/nwpu.py` |
| `parsers/base.py` (17 L) | **Dead code.** Byte-identical duplicate of `models.py`. | — | **None.** No importer anywhere in `src`, `scripts`, `tests`, `notebooks`. |
| `parsers/nwpu.py` (52 L) | Parse NWPU `(x1,y1),(x2,y2),class` text via a strict `fullmatch` regex; raises on any deviation. | `Path` → `list[BoundingBox]` | `splitting.py`, `yolo.py`, `ultralytics_backend.py`, `scripts/prepare_dataset.py` |
| `adapters/nwpu.py` (97 L) | Discover the three NWPU folders by case-insensitive `rglob` on directory name; pair positive images with same-stem annotations; mark the 150 negatives as backgrounds. Sorted → deterministic order. | `dataset_root: Path` → `list[DatasetItem]` (800) | `scripts/prepare_dataset.py`, notebook 02 |
| `audit.py` (183 L) | Integrity audit: PIL `verify()`, SHA-256 duplicate groups, box-coordinate validity, class-ID validity, per-class counts, relative-area size bucketing. | `items, parser, valid_class_ids` → report dict | `scripts/prepare_dataset.py`, notebook 01 |
| `config.py` (33 L) | Load `configs/datasets/<name>.yaml`, assert three sections present. | name + root → dict | **Notebook 01 only.** No `src`/`scripts` caller. |
| `reporting.py` (34 L) | Pretty-print audit dict; two `pd.DataFrame` wrappers. | report dict → stdout / DataFrame | **Notebook 01 only.** |
| `splitting.py` (365 L) | Build an (N, 11) multilabel presence matrix (10 classes + background), then two nested `MultilabelStratifiedShuffleSplit` passes (70 / 15 / 15, seed from config). Validates non-empty, no duplicate paths, no cross-split path overlap, backgrounds present in each split. Writes a CSV manifest with paths relative to `dataset_root`. | `list[DatasetItem]` → `{"train"/"val"/"test": [DatasetItem]}`; → manifest CSV | `manifests.py`, `yolo.py`, `scripts/prepare_dataset.py`, notebook 02 |
| `manifests.py` (60 L) | Reconstruct splits from the CSV. Keys on `image_path`/`annotation_path` columns; **ignores the `image_id` column entirely**. Asserts every referenced file exists, then revalidates the splits. | manifest CSV + `dataset_root` → `DatasetSplits` | `scripts/prepare_dataset.py`, `ultralytics_backend.py` |
| `yolo.py` (242 L) | NWPU→YOLO conversion (class_id − 1, xyxy→normalised cxcywh), copy images under collision-safe stems `{split}_{idx:04d}_{positive\|background}_{stem}`, write `dataset.yaml`, `export_manifest.csv` (source-image provenance), `export_summary.json`. Separate `validate_yolo_dataset()` re-parses every label file and checks stems, field counts, ranges, in-image containment, and expected totals. | splits + output root + 10 class names → on-disk YOLO tree + report | `scripts/prepare_dataset.py` |

### `src/detection/`

| Module | Actual responsibility | Real inputs → outputs | Callers |
|---|---|---|---|
| `config.py` (97 L) | `${VAR}` expansion that **hard-fails on any unset variable**; validate the 11 required training sections; `resolve_path`; `save_resolved_config` (strips `_`-prefixed keys). | YAML path → dict | all training/eval/predict scripts, `inference_config.py`, `ultralytics_backend.py` |
| `inference_config.py` (67 L) | Independent validator for the inference YAML (5 required sections + optional `ground_truth` block). Enforces `0 ≤ conf,iou ≤ 1`, `0 < iou_threshold ≤ 1`, `dataset_root` required iff `manifest` set. | YAML path → dict | `scripts/export_predictions.py` |
| `inference.py` (358 L) | The record layer. Four frozen dataclasses (`InferenceImageRecord`, `DetectionRecord`, `GroundTruthRecord`, `ComparisonRecord`); YOLO-label→pixel-xyxy parser; IoU; **greedy one-to-one GT matching** (`compare_predictions_to_ground_truth`, L138–220); referential-integrity validator; image discovery (sorted `rglob`); CSV writers. | records → `inference_images.csv`, `detections.csv`, `ground_truth.csv`, `prediction_comparison.csv` | `ultralytics_backend.py`, tests |
| `reporting.py` (58 L) | Runtime provenance: package versions, `git rev-parse HEAD`, platform, CUDA device, `RS_JOB_ID`. | project root → dict; `write_json` | `ultralytics_backend.py` |
| `ultralytics_backend.py` (508 L) | The only place Ultralytics is touched. `train_detector` (L56), `evaluate_detector` (L104, writes `overall_metrics.json` + `per_class_metrics.csv`), `export_predictions` (L255, the structured path), `predict_images` (L474, thin passthrough). | config + checkpoint → run directory | all four CLI scripts |

### `src/utils/`

| Module | Actual responsibility | Callers |
|---|---|---|
| `paths.py` (17 L) | Walk upward for a dir containing `configs`, `src`, `notebooks`. | all scripts |
| `archive.py` (30 L) | Zip-slip-safe extraction, skip-existing by default. | `scripts/prepare_dataset.py` |
| `job_logging.py` (145 L) | `@logged_cli` decorator: tee stdout/stderr to `${JOB_LOG_ROOT}/<cmd>/<ts_uuid>/run.log`, write `job.json` with status/exit code/duration, export `RS_JOB_ID`/`RS_JOB_DIRECTORY`. Silently degrades to no-op if neither env var is set. | all four CLI scripts |

### `scripts/` (entry points, all registered in `pyproject.toml`)

| Script | What it actually does |
|---|---|
| `prepare_dataset.py` (145 L) | extract-if-missing → audit (reusable from JSON) → split (reusable from manifest) → YOLO export (reused if non-empty and validating). `--force` re-runs all three. |
| `train_detector.py` (25 L) | config → `train_detector`, print `best.pt`. |
| `evaluate_detector.py` (37 L) | config + `--checkpoint`; refuses `evaluation.split=test` without `--confirm-test`. |
| `export_predictions.py` (27 L) | config only. No CLI overrides at all — by design (test `test_export_predictions_cli_uses_only_config`). |
| `predict_detector.py` (29 L) | config + checkpoint + `--source`; writes only Ultralytics' native output plus a metadata JSON. **Produces no structured CSVs.** |

### Configs

| File | Read by | Notes |
|---|---|---|
| `configs/detection/yolov8n_baseline.yaml` | prepare / train / evaluate / predict | `prediction.confidence: 0.25` (L83) |
| `configs/inference/ultralytics_export.yaml` | export-predictions | `inference.confidence: 0.05` (L26), `iou: 0.70` |
| `configs/datasets/nwpu_vhr10.yaml` | notebook 01 only | 1-based class map; paths point at a `shared_resources/` that does not exist at repo root |
| `configs/audits.yaml` | **nothing** | Dead config. No loader references it. |

### Tests (22 tests)

`tests/data/` (manifest round-trip, positive/background filename-collision avoidance),
`tests/detection/` (config env expansion, inference-config independence, record validation, GT
comparison statuses, YOLO-label parsing, backend export incl. the negative-image annotation guard and
the test-authorization guard), `tests/utils/` (job logging tee + failure path).

**Current state, run locally:** 4 of 8 files fail to collect (`iterstrat`, `ultralytics`, `torch` not
installed in this environment) and `tests/detection/test_inference_config.py::test_load_inference_config_is_independent_from_training`
**fails on merit**: it sets `EXPERIMENT_OUTPUT_ROOT` but the shipped YAML now interpolates
`${SHARED_RESOURCES_ROOT}` (`configs/inference/ultralytics_export.yaml:6`). The test is stale
relative to the config it loads. 13 pass.

---

## 2. Detector state

| Property | Value | Source |
|---|---|---|
| Variant | **YOLOv8n**, `yolov8n.pt` COCO-pretrained → fine-tuned | `configs/detection/yolov8n_baseline.yaml:50`, `training.pretrained: true` L64 |
| Trained on | NWPU VHR-10 YOLO export `nwpu_vhr10_yolo_seed42_v3`, train split | notebook 02 cell 33 output (`data=/content/datasets/processed/nwpu_vhr10_yolo_seed42_v3/dataset.yaml`) |
| Epochs / imgsz / batch | 50 / 640 / 16, patience 15, optimizer `auto` (resolved to AdamW by Ultralytics), warmup 3.0, `cos_lr=false`, `lrf=0.01`, `deterministic=true`, seed 42 | config L52–69; confirmed in notebook 02 cell 33 arg dump |
| Ultralytics / torch | 8.4.112 / 2.11.0+cu128, Tesla T4 | notebook 02 cell 33 output |
| Checkpoint selection | `best.pt` by val, epoch 49 | notebook 02 cell 39 |
| Eval thresholds | **`evaluate_detector` never passes `conf` or `iou` to `model.val()`** (`ultralytics_backend.py:119–131`) → Ultralytics defaults (conf ≈ 0.001, NMS IoU 0.7). Thresholds are therefore implicit, not configured. | code read |
| Prediction thresholds | Two different values live in the repo: `0.25` for `rs-predict-detector`, `0.05` for `rs-export-predictions`. NMS IoU `0.70` in both. `max_det` is never set → Ultralytics default 300. | configs |

### Measured numbers actually present in the repo

All numbers below come from **executed notebook outputs**, not from JSON on disk. No
`overall_metrics.json`, `per_class_metrics.csv`, checkpoint, or prediction CSV exists in this
working copy — those live in the Drive `shared_resources/experiment_outputs/` tree, which
`.gitignore` excludes.

**Validation, best epoch 49** — `notebooks/02_object_detection_baseline.ipynb`, cell 39 output:

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.9540 | 0.9358 | 0.9758 | 0.6345 |

**Held-out test (118 images / 523 instances)** — cell 43 + 44 output:

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 118 | 523 | **0.927** | **0.932** | **0.962** | **0.637** |
| airplane | 13 | 101 | 0.965 | 0.990 | 0.984 | 0.684 |
| ship | 9 | 48 | 0.974 | 0.790 | 0.869 | 0.572 |
| storage_tank | 4 | 70 | 0.851 | 0.986 | 0.981 | 0.592 |
| baseball_diamond | 27 | 56 | 0.979 | 0.964 | 0.991 | 0.715 |
| tennis_court | 16 | 70 | 0.936 | 0.986 | 0.994 | 0.735 |
| basketball_court | 13 | 29 | 1.000 | 0.880 | 0.958 | 0.653 |
| ground_track_field | 25 | 25 | 0.925 | 1.000 | 0.995 | 0.849 |
| harbor | 4 | 26 | 0.867 | 0.999 | 0.971 | 0.500 |
| bridge | 10 | 16 | 0.834 | 0.940 | 0.963 | 0.465 |
| vehicle | 13 | 82 | 0.941 | 0.782 | 0.913 | 0.608 |

Speed: 3.5 ms pre / 9.6 ms inference / 4.8 ms post per image (T4).

**This is a strong detector, and that is a problem for the project.** mAP50 0.96 with recall 0.93 on
test means the uncertain band the verification layer exists to triage is thin. At conf 0.05 the
proposal population will be dominated by easy true positives. Expect the B0 (calibrated confidence
only) baseline to be hard to beat — which is a legitimate finding, but it should be anticipated, not
discovered in week 4. Sample-size caveat: 4 test images carry all storage-tank and all harbor
instances; per-class test numbers for those two classes are noise.

### Artifact format produced by `rs-export-predictions`

Four CSVs in the Ultralytics run directory, plus Ultralytics' own `labels/` and rendered images:

- `inference_images.csv` — `image_id, dataset_id, source_image_path, source_split, image_width, image_height, prediction_count, ground_truth_count`
- `detections.csv` — `detection_id, image_id, predicted_class_id, predicted_class_name, confidence, xmin, ymin, xmax, ymax` (absolute px, clipped to image, `xyxy`)
- `ground_truth.csv` — `ground_truth_id, image_id, class_id, class_name, xmin, ymin, xmax, ymax`
- `prediction_comparison.csv` — `comparison_id, image_id, detection_id, ground_truth_id, iou, status` where status ∈ `true_positive | class_error | false_positive | false_negative`
- `resolved_config.yaml` + `inference_metadata.json` (git commit, package versions, CUDA device, conf/iou, match counts)

The join `detections ⋈ prediction_comparison ⋈ ground_truth ⋈ inference_images` reconstructs most of
the brief's `Proposal` dataclass. Missing fields: `dataset`, `obj_area_px`, `verdict` (the four
statuses map to accept/relabel/reject but are not named that), and a stable `proposal_id`.

---

## 3. Dataset state

### On disk

`docs/shared_resources/datasets/` — **untracked by Git** (`.gitignore` line `shared_resources/*`
matches at any depth).

| Item | Value |
|---|---|
| `raw/NWPU VHR-10 dataset/positive image set/` | 650 JPEG, all RGB, 644 distinct resolutions (range roughly 530×490 – 1100×900) |
| `raw/NWPU VHR-10 dataset/negative image set/` | 150 JPEG, all RGB |
| `raw/NWPU VHR-10 dataset/ground truth/` | 650 `.txt`, **3,896 boxes, 0 unparseable lines** (verified by re-running the repo's regex over every file) |
| `archive/NWPU VHR-10 dataset.zip` | source archive, kept |
| `manifests/nwpu_vhr10_split_seed42.csv` | 800 rows — see §5 for why it is unusable |
| HRRSD | **absent.** Not on disk, not in any config, not referenced by any code. |

Provenance per the dataset readme: Google Earth + Vaihingen (DGPF) crops, expert-annotated, axis-aligned
xyxy, class IDs 1–10. **No GSD metadata exists per-image**, and NWPU mixes two very different sources
(Google Earth colour ≈ 0.5–2 m; Vaihingen ≈ 0.08 m pan-sharpened). The brief's instruction to define
context-crop extent in metres is not currently satisfiable per-image from anything in the repo.

### Class taxonomy

Declared in three places, all consistent, all hard-coded to exactly 10:
`configs/datasets/nwpu_vhr10.yaml:10-20` (1-based), `configs/detection/yolov8n_baseline.yaml:33-43`
(list order = YOLO 0-based ID), `notebooks/02` cell 17 (`YOLO_CLASS_NAMES`).
Enforced by `src/detection/config.py:51-52`, `src/data/yolo.py:67-68`, `src/data/yolo.py:34`,
`src/data/splitting.py:22` (`number_of_classes=10` default).

**No cross-dataset label mapping exists** — no file, no dict, nothing. That is expected (HRRSD absent)
but it means §4.3 of the brief is entirely unstarted.

### Box-size distribution (computed directly from the 3,896 GT boxes)

| Class | n | median box area px² | median side px |
|---|---|---|---|
| airplane | 757 | 4,950 | 70 |
| storage_tank | 655 | 2,700 | 52 |
| vehicle | 598 | 1,856 | **43** |
| tennis_court | 524 | 3,382 | 58 |
| baseball_diamond | 390 | 6,086 | 78 |
| ship | 302 | 2,631 | 51 |
| harbor | 224 | 8,326 | 91 |
| ground_track_field | 163 | 60,352 | 246 |
| basketball_court | 159 | 6,540 | 81 |
| bridge | 124 | 20,095 | 142 |

Brief's Step-2 buckets:

| Bucket | Boxes | Share | Composition |
|---|---|---|---|
| `<32²` (<1,024 px²) | **103** | 2.6 % | storage_tank 77, vehicle 14, ship 8, airplane 4 |
| `32²–96²` | 3,188 | 81.8 % | all classes |
| `>96²` | 605 | 15.5 % | ground_track_field 163, bridge 110, harbor 96, baseball_diamond 96 |

**This contradicts a load-bearing assumption in the brief.** §4.2 states "vehicle crops in NWPU VHR-10
are ~10–20 px before upsampling". They are not: the median vehicle box is 43×43 px, and only 14 of 598
vehicle boxes fall below 32². The `<32²` column of the Step-2 encoder table would be computed over
~103 GT boxes (~2.6 % of the corpus), three-quarters of them storage tanks. Stratified sampling of
2,000 proposals cannot fill that cell meaningfully. See §7 Q1.

### How splits were generated

`create_dataset_splits` (`splitting.py:91`): deterministic given a fixed item order (which
`discover_nwpu_items` guarantees by sorting) and a fixed `iterstrat` version. Two nested
`MultilabelStratifiedShuffleSplit` calls with `random_state=42`. Cross-split image-path disjointness
is asserted (`splitting.py:239-249`). Manifest is written to disk sorted by `(split, image_id)`.

**Committed to Git: no.** The manifest lives only in the ignored `shared_resources/` tree, i.e. in
one Drive folder and this one working copy.

Three mutually inconsistent split states exist right now:

| Source | train | val | test | Notes |
|---|---|---|---|---|
| `manifests/nwpu_vhr10_split_seed42.csv` (on disk) | 560 (455 pos / 105 bg) | 120 (97/23) | 120 (98/22) | old `image_id` format, Windows path separators |
| Notebook 02 cell 14 (manifest it saved) | 562 (457/105) | 120 (98/22) | 118 (95/23) | |
| Notebook 02 cell 21 (YOLO export `_v3` the detector was trained on) | 562 | 120 | 118 | matches cell 14 |
| `configs/detection/yolov8n_baseline.yaml:23` expects | — | — | — | file `nwpu_vhr10_multilabel_split_seed42.csv`, **which does not exist** |

Same seed, same ratios, different results — so the on-disk manifest was produced by an earlier code
path or a different `iterstrat` build. It is *not* the split the checkpoint was trained on.

---

## 4. Gap analysis vs. WORK ORDER (§7)

| Item | Status | Evidence |
|---|---|---|
| **Step 0 — low-confidence regeneration** | **EXISTS** | `configs/inference/ultralytics_export.yaml:26` is already `0.05`; `rs-predict-detector` still uses `0.25` (`yolov8n_baseline.yaml:83`) but produces no structured output, so it is irrelevant to Step 0. |
| **Step 0 — GT matching (greedy, IoU-first, one-to-one)** | **PARTIAL** | `src/detection/inference.py:138-220`. Greedy by descending IoU with deterministic tie-break on IDs (L157-164); a GT absorbs at most one detection (L165-177); class-aware verdict at L178-182. Correct algorithm. Gaps: threshold defaults to `0.5` in the signature (L141) rather than being required from config; verdicts are named `true_positive`/`class_error`/`false_positive` not `accept`/`relabel`/`reject`; unmatched detections record `iou=None` rather than their best sub-threshold IoU (the brief's analysis wants that value); no IoU-0.7 sensitivity pass. |
| **Step 0 — `Proposal` record with all fields** | **PARTIAL** | Four CSVs, joinable. Missing `proposal_id` (there is `detection_id`), `dataset`, `obj_area_px`, `verdict`, `matched_gt_id` on the detection row itself, and `gt_class`. No Parquet/JSONL writer; CSV only (`inference.py:304-358`). |
| **Step 0 — committed deterministic split ID lists** | **MISSING** | Manifest is `.gitignore`d, stale, non-portable, and disagrees with the trained checkpoint. See §5.1. |
| **Step 1 — crops (square, 10 % pad, context k·max(w,h), edge-shift, JPEG cache)** | **MISSING** | No cropping code anywhere. |
| **Step 2 — encoder bake-off** | **MISSING** | No `open_clip`, `torch`-embedding, DINOv2, retrieval-purity, or same-image-exclusion code. `requirements.txt` has 7 packages, none of them an encoder. |
| **Step 3 — memory + retrieval, α/k sweeps** | **MISSING** | No embedding store, no SQLite, no matmul retrieval. |
| **Step 4 — baselines B0–B3, calibration, risk–coverage/AURC, conformal** | **MISSING** | No calibration, no selective-prediction metrics. |
| **Step 5 — VLM** | **MISSING** | — |
| **Step 6 — ablations** | **MISSING** | — |
| **Step 7 — Streamlit UI** | **MISSING** | — |
| **§4.1 leakage control (HRRSD memory / NWPU eval, OpenCLIP ablation)** | **MISSING** | HRRSD is entirely absent. Note: the repo does **not** do the reverse either — it does nothing retrieval-related. |
| **§4.3 explicit committed label mapping** | **MISSING** | — |
| **§4.4 split discipline (disjoint image IDs, seeded, committed)** | **PARTIAL** | Disjointness is enforced in code (`splitting.py:239-249`) and seeding exists (`splitting.py:70`), but the artifact is not committed and the `image_id` scheme collides — see §5.2. |
| **§4.5 no FAISS** | **EXISTS (by omission)** | FAISS is not a dependency. But `README.md:5,12` and `README.md:116` still promise FAISS. Docs are stale relative to the brief. |
| **§8 image-level bootstrap, per-class breakdown, ECE, asymmetric costs** | **MISSING** | Per-class detector metrics exist (`ultralytics_backend.py:147-179`); nothing else. |

**Infrastructure that already exists and should be reused rather than rebuilt:** env-var-driven config
loading with fail-fast on unset vars, `resolve_path`, `save_resolved_config`, run provenance
(`collect_runtime_metadata` records the git commit — keep this), job logging, the record-validation
pattern in `validate_inference_records`, and the dataset/export validators. These are good and cheap
to extend.

---

## 5. Blocking issues

Ranked by how badly each one invalidates a downstream measurement.

### 5.1 The committed split does not match the trained detector — and is not committed

`configs/detection/yolov8n_baseline.yaml:23` points at `nwpu_vhr10_multilabel_split_seed42.csv`.
That file does not exist. The file that *does* exist, `nwpu_vhr10_split_seed42.csv`, has a different
train/val/test partition (560/120/120) from the one the checkpoint was trained on (562/120/118).

Consequence: running `rs-prepare-detector` today takes the `reuse_manifest` branch only if the
configured filename happens to be present; otherwise it silently **generates a fresh split**
(`scripts/prepare_dataset.py:103-115`), and that split will differ from the checkpoint's. Any case
memory built from the "train" side of a regenerated split will contain images the detector saw as
val/test — or worse, evaluation will run on images the detector trained on. Every coverage and
risk–coverage number computed on top of that is void, silently.

This is the single thing that must be fixed before any other work. The split ID lists must be
committed to Git (they are 800 lines of text), and the checkpoint that goes with them must be
identified.

### 5.2 The on-disk manifest is not loadable on Linux and has colliding IDs

Two independent defects in `manifests/nwpu_vhr10_split_seed42.csv`:

1. **Windows path separators.** Rows read `NWPU VHR-10 dataset\positive image set\002.jpg`.
   `load_split_manifest` does `dataset_root / row["image_path"]` (`manifests.py:41`), which on POSIX
   yields a single filename containing backslashes → `FileNotFoundError` at L48. The manifest cannot
   be loaded on Colab or on this Linux box.
2. **`image_id` collides.** The manifest's `image_id` is the bare stem, so positive `002.jpg` and
   negative `002.jpg` share `image_id=002`. **150 of 800 rows are duplicated IDs; 64 of those 150
   pairs straddle two different splits** (e.g. `002` → positive/test *and* negative/train). Any
   downstream code keyed on `image_id` would join test images to train rows. The current
   `save_split_manifest` (`splitting.py:321-322`) fixes this by prefixing `positive_`/`background_`;
   the on-disk file predates that fix. `load_split_manifest` never reads the column, so nothing has
   caught it.

### 5.3 The test-split guard is bypassable, and the shipped config bypasses it

`ultralytics_backend.py:295-298` refuses test images only when `source.split == "test"` **or** a
manifest maps some image to test. With `source.manifest: null` — which is what
`configs/inference/ultralytics_export.yaml:12` ships — `source_splits` is empty and
`source.split: external` passes the check. The shipped config then points `source.path` at
`.../datasets/raw/NWPU VHR-10 dataset` (L9), i.e. **all 800 images, test included**, and
`discover_inference_images` recurses (`inference.py:294-298`) and picks up every one.

The default configuration in the repo runs inference over the held-out test set with no warning.
Nothing downstream records which proposals came from test images (`source_split` is written as the
literal string `external` for all 800). This is exactly the class of error §4.4 exists to prevent.

### 5.4 `image_id` is positional and unstable

`ultralytics_backend.py:353` builds `image_id = f"{dataset_id}_{index:06d}_{stem}"` where `index` is
the position in the discovered file list. Re-running on a subset, or after adding a file, renumbers
everything. IDs are not comparable across runs, so proposals cannot be traced back to a specific
source image except via `source_image_path`, which embeds absolute machine-specific paths
(`/content/...` vs local). Case memory and prediction tables written in different runs will not join.

### 5.5 Thresholds and category boundaries are baked into module bodies

Against §11 ("every threshold, seed, and path lives in config"):

- `inference.py:141` — `iou_threshold: float = 0.5` as a function default.
- `audit.py:130-135` — size categories at relative area 0.01 / 0.05, hard-coded.
- `yolo.py:34`, `ultralytics_backend.py:232` — `range(1, 11)` class-ID validity.
- `yolo.py:67-68`, `detection/config.py:51-52` — "exactly 10 classes" assertions.
- `splitting.py:22` — `number_of_classes: int = 10` default.
- `evaluate_detector` never sets `conf`/`iou` for validation, so the reported mAP depends on
  Ultralytics' internal defaults, which changed across 8.x releases.
- `max_det` is never configured; at conf 0.05 the Ultralytics default of 300 silently truncates.

The 10-class assumptions are the expensive ones: adding HRRSD (13 classes) touches six files.

### 5.6 The detector is too good for the stated experiment

mAP50 0.962 / recall 0.932 on test. The verification layer's value comes from the uncertain band;
here that band is narrow, and at 118 test images / 523 instances the *entire* evaluation population
is small for image-level bootstrap CIs (§8 requires resampling at the image level — n=118 clusters).
This does not invalidate anything, but it caps the achievable effect size and the CI width. Decide
now whether NWPU-test is the evaluation set or whether the evaluation population needs to be larger.

### 5.7 Smaller, real

- `parsers/base.py` is a dead byte-identical duplicate of `models.py`. Delete it before someone
  imports the wrong `BoundingBox`.
- `configs/audits.yaml` is read by nothing.
- `notebooks/00_colab_bootstrap copy.ipynb` is an accidental duplicate.
- `tests/detection/test_inference_config.py::test_load_inference_config_is_independent_from_training`
  fails against the config it ships with.
- `README.md:5,12,116` still advertises FAISS; the brief removes it (§4.5).
- The notebooks re-implement `prepare_yolo_dataset_v2`, `convert_box_to_yolo`, and `format_yolo_box`
  inline (notebook 02 cells 17–18) rather than importing `src/data/yolo.py`. The trained detector was
  produced by the notebook copy, not the library. The two agree today; nothing enforces that.
- `_ground_truth_annotation_path` (`ultralytics_backend.py:202-205`) guards against matching a
  negative image to a same-stem positive annotation **only for `format: nwpu`**. For `format: yolo`
  the stem-fallback at L213 has no such guard.

---

## 6. Where the brief and the repository disagree

Surfaced rather than silently resolved, per §11.

1. **Small-object premise.** §4.2 says NWPU vehicle crops are ~10–20 px. Measured: median 43 px,
   with only 103 of 3,896 boxes below 32². The go/no-go gate as specified will be evaluated on a
   nearly empty cell.
2. **Contamination design.** §4.1 prefers memory-from-HRRSD, eval-on-NWPU. The repo has no HRRSD at
   all, so the "reverse" the brief asks to flag does not exist — but neither does the intended
   direction. Acquiring and preparing HRRSD (21,761 images) is itself a multi-day task inside an
   8-week budget.
3. **GSD in metres.** §4.2 requires context-crop extent in metres or GSD-normalised. NWPU ships no
   per-image GSD and mixes Google Earth with Vaihingen at very different resolutions. Not currently
   derivable from anything in the repo.
4. **FAISS.** README promises it; brief forbids it. Brief wins; README needs a one-line fix.
5. **Confidence floor.** Already satisfied in the export path (0.05). The brief's "if the current
   predictions were generated at 0.25, they must be regenerated" does not apply to
   `rs-export-predictions`.

---

## 7. Questions for the human

**Q1 — the Step-2 gate, given the real size distribution.** The `<32²` bucket holds 103 boxes (2.6 %),
77 of them storage tanks. Options: (a) keep the brief's fixed bucket edges and report the `<32²` cell
as under-powered; (b) re-cut the buckets at NWPU's own quantiles (e.g. `<2k`, `2k–8k`, `>8k` px²) and
say so; (c) scope the small-object question to a per-class vehicle/ship/storage-tank analysis instead
of a size analysis. I recommend (b) plus an explicit note, but this changes the headline gate table
and is your call.

**Q2 — HRRSD: in or out?** The entire §4.1 leakage-control design, the open-set novelty test (§4.3),
and a memory large enough for the growth curve (§6, 10k cases) depend on it. NWPU alone gives 3,896
GT boxes total; a case memory built from NWPU-train tops out around 2.8k. If HRRSD is out, the
memory-growth headline figure caps at ~2k and the RemoteCLIP-contamination ablation loses its
contrast. If it is in, week 1 has to absorb a 21k-image download, parse, and audit. Which?

**Q3 — which split and which checkpoint are canonical?** Three partitions exist (§3). I need one
declared authoritative. Cleanest path: regenerate the split with the current code, commit the ID
lists to Git, and retrain — 50 epochs on a T4 is under an hour by the notebook's own timings.
Alternative: locate the exact `_v3` export manifest on Drive and commit that. Retraining is the
lower-risk option; confirm before I touch anything.

**Q4 — evaluation population.** 118 test images / 523 instances is small for image-level bootstrap
CIs. Do you want (a) NWPU-test as-is, accepting wide intervals; (b) a merged val+test evaluation set
with memory built only from train; or (c) NWPU entirely as evaluation with memory from HRRSD (depends
on Q2)?

**Q5 — where does the case memory come from, given NWPU is the only dataset?** If memory and
evaluation both come from NWPU, the image-level disjointness rule (§4.4) means memory is built from
train-split *proposals*, which the detector has already fit. Retrieval purity measured on
detector-memorised training crops is optimistic. Do you want memory built from val instead, accepting
a ~500-case memory?

**Q6 — proposal record format.** Brief says Parquet or JSONL; the repo writes CSV and has no
`pyarrow`. Extend the existing CSV writers, or add `pyarrow` and switch to Parquet? CSV is adequate
at 10⁴ rows and adds no dependency; I'd keep CSV unless you want Parquet for the report.

**Q7 — `verdict` naming.** Rename the existing `true_positive`/`class_error`/`false_positive` statuses
to the brief's `accept`/`relabel`/`reject`, or add a derived column and leave the detector-diagnostic
names intact? Renaming breaks the four existing backend tests; adding a column does not.

---

## 8. Recommended first actions (for your approval, not yet executed)

1. Fix §5.1 + §5.2 + §5.3 as one change: regenerate the split with current code, commit the ID lists
   under `configs/splits/`, make `source.split` mandatory and the test guard unconditional, retrain.
2. Extend the existing export path with `dataset`, `obj_area_px`, stable content-derived
   `proposal_id`, and best-IoU-for-unmatched — no new module, no new abstraction.
3. Then Step 1 (crops) and the Step-2 gate, with bucket edges settled by Q1.

Nothing above has been implemented. No code was written or modified in this session.
