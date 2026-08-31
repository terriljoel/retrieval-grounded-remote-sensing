# BRIEF AMENDMENT 01

Supersedes the named sections of `docs/PROJECT_BRIEF.md`. Written after `docs/REPO_ANALYSIS.md`
(commit `85f059c`). Commit alongside the brief; do not edit the brief in place — the diff is the
record of what changed and why.

---

## A. Corrections to the brief

### A.1 §4.2 small-object premise — WRONG, retracted

The brief stated NWPU vehicle crops are ~10–20 px and made that the basis of the week-1 go/no-go
gate. Measured from the 3,896 GT boxes: median vehicle box is **43 px**; only 103 boxes (2.6%) fall
below 32², of which 77 are storage tanks. The error came from reasoning from GSD (0.5–2 m) alone and
ignoring that NWPU mixes Google Earth with Vaihingen pan-sharpened crops at 0.08 m.

**Consequence for the gate**: the *question* stands — can embedding retrieval discriminate at all —
but the *size-stratified framing* is under-powered. See §B.1 for the replacement.

**New action, 10 minutes, do it first**: plot the vehicle (and storage-tank, and ship) box-area
histogram. If bimodal, NWPU contains a hidden sensor-domain split (Google Earth vs. Vaihingen), and
retrieval results will partly measure source-sensor similarity rather than semantic similarity. If
bimodal, record a `source_sensor` field on every proposal and stratify all retrieval reporting by it.

### A.2 §4.2 GSD-in-metres requirement — retracted as unsatisfiable

NWPU ships no per-image GSD and mixes two sources at very different resolutions. Define context-crop
extent in **pixels as a multiple of box size** (`k × max(w,h)`) and state the limitation explicitly in
the report. Revisit only if HRRSD ships usable per-image resolution metadata.

### A.3 §5 confidence floor — already satisfied

`configs/inference/ultralytics_export.yaml:26` is already `0.05`. The brief's instruction to
regenerate predictions does not apply to `rs-export-predictions`. The `0.25` in
`yolov8n_baseline.yaml:83` belongs to `rs-predict-detector`, which emits no structured output and is
out of scope.

### A.4 NEW §4.6 — detector strength is an experimental variable, not a fixed input

The trained YOLOv8n reaches mAP50 0.962 / R 0.932 on NWPU test. That leaves almost no uncertain band,
caps the achievable effect size, and makes B0 (calibrated confidence only) very hard to beat.

**This is a framing error in the original brief, not just a nuisance.** An annotation assistant is a
low-data-regime tool. A detector at 0.96 mAP is the *end state* of an annotation campaign, not the
start. The realistic use case is 200 labelled images and a mediocre detector.

**Requirement**: train a **detector family** at train-split fractions **{25%, 50%, 100%}**, stratified
subsample, same seed, same hyperparameters, same eval set. Run the full evaluation pipeline against
each. ~1 h per run on a T4; no new code beyond a subsample flag in the split loader.

The headline result becomes two-dimensional:

> *Coverage @ 1% risk as a function of detector strength — retrieval-grounded verification adds X
> points at mAP 0.80, Y at 0.90, Z at 0.96.*

If time compresses, drop to {25%, 100%}. Do not drop to a single detector.

### A.5 NEW §4.7 — the brief omitted a calibration split

Temperature scaling of detector confidence, fitting the B2 logistic gate, and conformal threshold
selection all require held-out data that is neither training nor evaluation data. The brief specified
only train/eval. **Three-way split is mandatory**: `train` (detector) / `calib` (gate + conformal) /
`eval` (reported numbers). Nothing fitted on `calib` may be refitted on `eval`, and no number computed
on `calib` appears in the report except as a calibration diagnostic.

---

## B. Decisions on the seven open questions

### B.1 Q1 — Step-2 gate stratification: **per-class primary, size secondary**

The size buckets are under-powered (n=103 in `<32²`). Restructure the gate table:

**Primary table — per class** (10 rows, all populated, n ≥ 124 each). This is what actually matters:
retrieval either discriminates airplanes from storage tanks or it does not.

**Secondary table — size**, reported twice in the same block:
- COCO-standard edges (`<32²`, `32²–96²`, `>96²`) for comparability with the CV literature, with the
  `<32²` cell explicitly annotated `n=103, under-powered`.
- Quantile tertiles of the NWPU box-area distribution, edges stated numerically.

**Gate criterion, restated**: compare each per-class top-5 purity against that class's prior in the
memory. If purity ≈ prior for a class, retrieval carries no signal for it — scope the claim to the
classes where it does and say so. The gate is a scoping instrument, not a pass/fail.

### B.2 Q2 — HRRSD: **IN, subsampled**

Not all 21,761 images. Take a **seeded random subset of ~4,000 images from HRRSD's train split**,
committed as an ID list under `configs/splits/hrrsd_memory_subset.txt`. At ~2.6 boxes/image that
yields ~10,000 cases — exactly the ceiling the memory-growth curve needs.

This unlocks four things NWPU alone cannot provide: the RemoteCLIP-contamination ablation (HRRSD is in
DET-10, NWPU is not), the open-set novelty test (HRRSD's crossroad / T-junction / parking lot are
unseen classes), a memory large enough for the growth curve, and memory/evaluation separation at the
dataset level.

Budget ~1 day: subset download, VOC-XML parser + adapter following the existing `parsers/nwpu.py` and
`adapters/nwpu.py` patterns, audit run. The existing pipeline is reusable; the cost is the six
hard-coded 10-class assertions (`yolo.py:34,67-68`, `detection/config.py:51-52`, `splitting.py:22`,
`ultralytics_backend.py:232`), which must become config-driven.

**Do not train the detector on HRRSD.** HRRSD is memory-only. The detector stays NWPU-only. This keeps
the detector family clean and avoids a second training axis.

### B.3 Q3 + Q4 — split and checkpoint: **re-split, retrain, do not salvage**

Salvaging any of the three existing partitions costs more than regenerating and carries permanent
doubt. Regenerate with current code and commit.

**New NWPU partition: 50% train / 20% calib / 30% eval** (≈ 400 / 160 / 240 images).

Rationale: 400 training images is ample for the 100%-fraction detector and is deliberately toward the
realistic-annotation regime (§A.4); 160 calibration images supports temperature scaling and conformal
threshold selection; 240 evaluation images (≈ 1,170 GT instances) gives n=240 clusters for image-level
bootstrap CIs, versus the 118 the current split would allow. Stratification stays multilabel via
`iterstrat`, seed 42, backgrounds present in every split.

Commit the three ID lists as plain text under `configs/splits/`. They are 800 lines total. Never
`.gitignore` them again.

Retrain the family (§A.4) against this partition. Discard the current checkpoint.

### B.4 Q5 — memory containing train-split images is realistic, not a leak

The disjointness rule that matters is **evaluation images must not appear in memory**. Memory
containing images the detector trained on is exactly the real workflow: those images were annotated,
became cases, and were used for training.

**Memory composition**: HRRSD subset (~10,000 cases) + NWPU `train` + NWPU `calib` cases. Evaluation:
NWPU `eval` only.

**One confound to document, not fix**: for NWPU-origin cases, detector confidence and retrieval
similarity are correlated through shared training data. This inflates B0 and B1 together, so the B1−B0
delta is roughly unaffected, but state it in limitations.

### B.5 Q6 — **keep CSV**

At 10⁴ rows CSV is adequate and adds no dependency. Extend the existing writers in
`inference.py:304-358`. Revisit only if a single table exceeds ~10⁶ rows.

### B.6 Q7 — **add a derived `verdict` column; do not rename**

`true_positive` / `class_error` / `false_positive` / `false_negative` are correct terminology in
detection evaluation. `accept` / `relabel` / `reject` are correct terminology in the annotation
workflow. These are two vocabularies over one fact. Add `verdict` as a derived column with the mapping
committed in config. Four existing tests stay green.

---

## C. Blocking issues — disposition

| # | Issue | Disposition |
|---|---|---|
| 5.1 | Split ≠ checkpoint, split uncommitted | **BLOCKER.** Resolved by §B.3 re-split + retrain. |
| 5.3 | Test-guard bypassable; shipped config runs over all 800 images incl. test | **BLOCKER, worst of the set** — it fails silently. Make split membership **mandatory**: every inference image must resolve to a committed split ID list, and the run must **abort** on any image that does not, or on any `eval` image unless `--confirm-eval` is passed. Delete `source.split: external` as a legal value. |
| 5.4 | `image_id` positional and unstable | **BLOCKER.** Replace with content-derived stable ID: `f"{dataset}_{subset}_{stem}"`, asserted unique at construction. Same scheme for `proposal_id`: `f"{image_id}_{det_index:03d}"`. Without this no two runs join. |
| 5.2 | Manifest Windows-separator + colliding IDs | Subsumed by §B.3. Current `save_split_manifest` already prefixes `positive_`/`background_`. Add a POSIX-separator assertion on write. |
| 5.5 | Thresholds in module bodies | **Partial fix now**: `iou_threshold` (required from config, no default), `max_det` (set explicitly), `evaluate_detector` conf/iou (pass explicitly so mAP does not float with Ultralytics releases). Defer the 10-class assertions to HRRSD ingestion, where they must become config-driven anyway. |
| 5.7 | Notebooks re-implement `src/data/yolo.py` inline | **Not tidiness.** The checkpoint came from notebook code, not library code. Since retraining anyway, make notebook 02 import from `src`. Add a test that the notebook's YOLO export path is the library one. |
| 5.7 | Dead `parsers/base.py`, dead `configs/audits.yaml`, duplicate notebook, stale `test_inference_config`, README FAISS lines | Tidiness. One batch, ~30 min, lowest priority. |

---

## D. SESSION 2 WORK ORDER

Execute in order. Phases A and B must complete before any retrieval code is written. Report at each
phase boundary; do not chain phases without checking in.

### Phase A — Foundation (blockers)

- **A0** Vehicle/storage-tank/ship box-area histograms (§A.1). 10 minutes. If bimodal, add
  `source_sensor` to the proposal schema before proceeding.
- **A1** Stable content-derived `image_id` and `proposal_id` (§C 5.4). Assert uniqueness at
  construction.
- **A2** Regenerate the NWPU split 50/20/30 with current code. Commit three ID lists to
  `configs/splits/`. POSIX separators asserted on write.
- **A3** Make split membership mandatory in the inference path; abort on unresolvable images; remove
  `external` as a legal `source.split`; require `--confirm-eval` for the eval split (§C 5.3).
- **A4** Config-drive `iou_threshold`, `max_det`, and `evaluate_detector` conf/iou (§C 5.5).
- **A5** Point notebook 02 at `src/data/yolo.py`; add the regression test (§C 5.7).
- **A6** Retrain the detector family at {25%, 50%, 100%} train fractions (§A.4). Record mAP50 /
  mAP50-95 / P / R per fraction on `calib` and `eval`.
- **A7** Tidiness batch.

### Phase B — Step 0 completion

- **B1** Extend the export path with `dataset`, `obj_area_px`, `verdict` (derived, §B.6), `gt_class`
  and `matched_gt_id` on the detection row, and best-sub-threshold IoU for unmatched detections. No
  new module; extend `inference.py`.
- **B2** IoU-0.7 sensitivity pass alongside the 0.5 primary.
- **B3** Run export for all three detectors × {calib, eval}. Six proposal tables.

### Phase C — HRRSD ingestion (§B.2)

- **C1** Seeded ~4,000-image subset from HRRSD train; commit ID list.
- **C2** VOC-XML parser + adapter mirroring the NWPU pattern; audit run.
- **C3** Config-drive the six 10-class assertions.
- **C4** Committed label-mapping YAML, NWPU ↔ HRRSD, with crossroad / T-junction / parking lot flagged
  `open_set: true`.

### Phase D — Steps 1–2 (unchanged from brief, gate restructured per §B.1)

- **D1** Crops (brief §6).
- **D2** Encoder bake-off with the per-class-primary table.

---

## E. Unchanged

Everything in the brief not named above stands: the selective-prediction reframe (§3), FAISS removal
(§4.5), the crop specification (§6), baselines B0–B3 (§7 Step 4), VLM-only-on-escalated (§7 Step 5),
the six ablations (§7 Step 6), the reporting requirements (§8), the honest novelty audit (§9), and the
architecture and feature freezes (§10).

The timeline slips by roughly one week against §10 to absorb Phase A and Phase C. Week 3's
architecture freeze moves to end of week 4. **The week-6 feature freeze does not move** — absorb the
slip out of Phase D onward, not out of the ablations. The ablations are where the grade is.
