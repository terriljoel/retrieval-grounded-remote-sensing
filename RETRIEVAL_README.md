# Retrieval + verification half — a guide for the detector side

For someone who built the detector but hasn't seen this half. No background
in embeddings or retrieval assumed.

## 1. What this does

Your model finds objects and says what they are. Mine decides which of those
findings a human still needs to double-check — by comparing each detection
against ~9,300 objects a human has already verified, and asking: does this
look like the things we already know are real? Confident match → auto-accept.
No match → a human sees it. Uncertain middle → a second AI looks first.

## 2. The pipeline

```
YOLO box + confidence
        |
  two crops per box (tight + wide)
        |
  RemoteCLIP: crop -> 768 numbers
        |
  compare against ~9,300 verified cases
        |
  top-5 most similar verified cases
        |
  logistic regression on 11 numbers
        |
  accept  /  escalate to human  /  ask a second AI
                                        |
                                  VLM reads the crop + the 5 similar cases
```

**YOLO box + confidence.** Your output — box, class, confidence — is the
start of everything below. Read via `src/detection/inference.py`'s
`detections.csv`/`inference_images.csv`; joined to our shared image IDs in
`src/retrieval/export_join.py`.

**Two crops per box.** A tight square on just the object, and a wider one
(4× the box) that includes surroundings — "what is this" vs "does the
context make sense." `src/retrieval/crops.py`.

**RemoteCLIP → 768 numbers.** A neural net trained on satellite imagery
(not ordinary photos) turns a crop into 768 numbers. Similar crops get
similar numbers — everything downstream depends on this holding.
`src/retrieval/encoders.py`, `src/retrieval/embed.py`.

**Compare against ~9,300 verified cases.** A database of past detections a
human already checked. Every new crop's 768 numbers get compared against
every case in it. `src/retrieval/memory.py`.

**Top-5 most similar cases.** The five closest matches, kept separately as
"supports this being real" and "contradicts it." `src/retrieval/gate.py`,
`src/retrieval/fuse.py`.

**Logistic regression on 11 numbers.** Best-match similarity, top-5 class
agreement, supporting-vs-contradicting margin, box geometry, sensor type. A
small classifier combines these into one score. `scripts/run_baselines.py`
(`build_features`), fit in `scripts/run_b0.py`.

**Accept / escalate / ask a second AI.** Above a threshold, auto-accept;
below it, a human sees it. `src/retrieval/gate.py`, `src/retrieval/conformal.py`;
threshold live in `app/gate_explorer.py`.

**VLM reads the crop + the 5 similar cases.** For the uncertain middle, a
second AI model gives its own yes/no, which can move a detection back to
accept or escalate. `scripts/run_b3_vlm.py`, `src/retrieval/nim.py`.

## 3. What we found

| Result | Number | Status |
|---|---|---|
| Retrieval finds the right class | P@5 = 0.991 vs 0.100 chance | Real (`reports/gate/GATE_REPORT.md`) |
| Spots an object class never seen before | AUROC 0.999, 0.2% false-flag rate | Real (`reports/phase1/OPEN_SET.md`) |
| Showing the AI real evidence genuinely helps | true 84.5% vs no-evidence 75.9% vs fake-evidence 72.5% (coverage at a 1% error budget) | Real (`reports/phase1/b3_vlm_*.csv`) |
| Down-weighting unverified cases protects against bad labels | at 20% corrupted labels: 29.2% → 57.5% (same budget) | **Synthetic only, not tested on real data** |
| Just trusting your detector's confidence beats our gate | confidence AUROC 0.976 vs our gate's 0.788, real detections | Real — the headline problem, below |
| Can't tell "no object" from "object there, box wrong" | bad box: −0.015 similarity; empty crop: −0.078 — 5× the gap | Real (`reports/gate/GATE_REPORT.md` §6) |

**The main finding — sim-to-real collapse and recovery.** Our gate was
trained entirely on made-up examples (clean boxes, shifted boxes, blank
crops), because we didn't have your real output yet. Run on your actual
detections: on the harm that matters most (a false detection on an empty
tile), the gate caught it only 0.38% of the time at a strict budget — your
detector's raw confidence number alone caught it 55.1% of the time.
Retraining the gate on just 919 of your real boxes recovered almost all of
that: 0.38% → 28.6%, close to matching your confidence number outright.
**A gate trained only on made-up examples does not work on real detector
output — it needs real examples.**

## 4. Things worth knowing about the detector side

Found while joining our two halves — shared as findings, not criticism.

- **The split files disagree with each other.** At least four "same"
  train/val/test lists exist in this project; two disagree on 335 of 800
  images, one mislabels 97 of 118 test images as train/val. We now key
  everything on one fixed file
  (`configs/splits/nwpu_vhr10_multilabel_split_seed42_fixed.csv`) and never
  read a split from anywhere else — worth checking which file your own
  numbers used.
- **The export we tested covers all 800 images**, no split column, and a
  very low confidence floor (down to 0.05 — closer to "everything
  considered" than a final output). We re-derived which image is which by
  re-scanning the source folder in the export's own order. Works, but
  fragile — an export with `inference_images.csv` and a split column would
  remove a real failure point.
- **NWPU's ground truth misses real objects.** One detection — confidence
  0.940, `ground_track_field` — scores as strongly "real" as any verified
  positive but has no matching ground-truth box. The object is genuinely
  there; the dataset never labelled it. Some of what looks like your
  detector's false-positive rate here is the dataset's fault.
- **Earlier checkpoints would genuinely help us.** Our one real-data test
  above had only 919 boxes, all from your final, fairly-confident
  checkpoint. A less-trained checkpoint makes more mistakes — exactly what
  our gate needs to see to learn a real boundary. If you still have earlier
  checkpoints, running them over the training images would give us much
  richer real errors to train on.

## 5. How to run it

```
export NIM_API_KEY='nvapi-...'
streamlit run app/gate_explorer.py
```

Pick an image, pick a detection, "Find similar cases," "Ask the VLM," see
the verdict. Four cases worth clicking through (`docs/UI_TEST_CASES.md` has
the full list):

- **A false positive on an empty tile** — correctly escalated.
- **A confident, correctly-boxed detection that still gets escalated** —
  the gate is currently too strict, not too loose.
- **The NWPU annotation-gap case** above — real object, no ground truth,
  correctly sent to a human either way.
- **Two classes that look alike** (tennis court / basketball court) — where
  the "5 similar cases" view helps a human most.

## 6. Where to look for more

- `reports/FINDINGS.md` — every result, numbered, tagged real vs
  synthetic-only.
- `reports/phase1/baseline_summary.csv` + `coverage_by_harm.csv` — the two
  simple baselines (nearest-neighbour vote, logistic regression) the gate is
  measured against. (`BASELINES.md`, the narrative write-up of these, was
  lost in an unrelated data-loss incident and hasn't been rebuilt yet —
  these CSVs are the real numbers behind it.)
- `reports/gate/GATE_REPORT.md` — the first test of whether retrieval works
  at all, before anything else was built on it.

For the detector half, see [`README.md`](README.md).
