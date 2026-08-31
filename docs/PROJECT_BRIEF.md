# PROJECT BRIEF — Human-in-the-Loop Remote Sensing Annotation Assistant

> **How to use this file.** Place at `docs/PROJECT_BRIEF.md` in the repo. In your first Claude Code
> session, say: *"Read `docs/PROJECT_BRIEF.md` in full, then execute the TASK FOR THIS SESSION
> section. Do not write code until you have delivered the repo analysis and I have responded."*

---

## 0. TASK FOR THIS SESSION

**Do not write or modify any code in this session.** Your first output is an analysis document, not an
implementation.

1. Inspect the entire repository. Read the actual source — do not infer behaviour from filenames,
   READMEs, or docstrings. Where planning documents contradict the implementation, the implementation
   is the source of truth.
2. Produce `docs/REPO_ANALYSIS.md` containing:
   - **Inventory**: every module, its actual responsibility, its real inputs/outputs, its callers.
   - **Detector state**: which YOLOv8 variant, trained or pretrained, on what split, with what
     hyperparameters, at what confidence/NMS thresholds, producing what artifact format. Report the
     actual mAP/PR numbers found in the repo, with the file they came from.
   - **Dataset state**: which datasets are present on disk, their splits, class taxonomies, whether a
     label mapping between taxonomies exists, image counts, and how splits were generated (seed?
     deterministic? committed to disk?).
   - **Gap analysis**: for each item in the WORK ORDER (§7) below, mark `EXISTS` / `PARTIAL` /
     `MISSING`, citing the file and line where it exists.
   - **Blocking issues**: anything in the current code that would invalidate downstream measurements.
     Be specific and blunt. Split leakage, non-deterministic ordering, thresholds baked in as magic
     numbers, predictions saved without the metadata needed to trace them back to source images.
   - **Questions for the human**: anything genuinely ambiguous. Do not guess and proceed.
3. Stop. Wait for my response before implementing anything.

**Explicitly do not**, in this session or the next: refactor working code, add abstraction layers,
introduce a plugin/registry/factory pattern, add a web frontend, add Docker, add a CI pipeline, add an
ORM, or install a vector database. Every one of these is a scope leak. The project is 8 weeks and its
grade lives in the evaluation, not the architecture.

---

## 1. Project identity

- **What**: M.Sc. Deep Learning course project, TU Braunschweig, IGP. Team of two.
- **Supervisors**: Dr.-Ing. Mehdi Maboudi, Dr. Pedro Achanccaray.
- **Duration**: ~8 weeks. Hard constraint.
- **Goal**: a technically strong, realistic AI engineering project — for course evaluation, portfolio,
  and ML engineering interviews. **Not** a publication. Novelty is not the objective; measurable
  rigour is.
- **System**: a verification layer that sits on top of any object detector and decides, per proposal,
  whether a human needs to look at it.

Pipeline:

```
YOLOv8 → proposals → object crop + context crop → embeddings
       → retrieval over verified case memory → calibrated gate
       → {auto-accept | auto-reject | escalate} → VLM (escalated only)
       → human decision → case memory grows
```

---

## 2. History (why the design is what it is)

The original proposal was "explainable object detection": YOLO → FAISS retrieval → VLM → natural
language explanation. The professor rejected it on eight grounds: no real application, no
justification for semantic retrieval, no justification for FAISS, no justification for the VLM,
questionable 8-week feasibility, no answer to "why not just use detector confidence", no clarity on
what retrieval stores, and no differentiation from standard detection pipelines.

The project was re-scoped to **annotation assistance**. That fixed the motivation problem. It did
**not** fix the measurement problem, which is addressed in §3.

---

## 3. THE CENTRAL DESIGN DECISION — read this before anything else

The revised proposal claims: *"retrieval-grounded VLM assistance reduces annotation effort while
maintaining annotation quality."*

**That claim is untestable by this team** and must not drive the implementation. Testing it requires
annotators, a timing protocol, a control group, and blinding. Two students who both know the
hypothesis cannot produce a credible timing study, and a weak one is worse than none — AI assistance
is documented to reduce annotation time while inducing silent semantic drift.

**The claim is replaced by selective prediction / learning-to-defer.** Every proposal is routed to
`auto-accept` / `auto-reject` / `escalate`. Ground truth exists for every proposal, so every routing
decision is scorable offline with zero humans involved.

The headline claim becomes:

> *At a 1% false-accept rate, retrieval-grounded verification auto-resolves X% of proposals;
> calibrated detector confidence alone auto-resolves only Y%.*

**Primary metric**: risk–coverage curve, summarised by AURC, one line per pipeline.
**Secondary**: coverage @ 1% and @ 5% false-accept rate.

Annotation *time* is reported only as a derived quantity, from a cost model
`T = n_verify·t_v + n_correct·t_c + n_draw·t_d` with `t_*` taken from published annotation-cost
literature (ILSVRC box drawing ≈ 30 s/box; verification is far cheaper — Papadopoulos et al.,
CVPR 2016), never from a stopwatch. State the constants and their source.

Framework references: El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017 (selective classification);
Mozannar & Sontag 2020 (learning to defer); Angelopoulos & Bates (conformal prediction).

---

## 4. Hard technical constraints

### 4.1 Dataset contamination — verify this in the repo

**RemoteCLIP was trained on HRRSD.** Its DET-10 pretraining collection is DOTA, DIOR, **HRRSD**,
RSOD, LEVIR, HRSC plus four UAV datasets. HRRSD contributes 21,761 images / 13 classes / 57,137
boxes. **NWPU VHR-10 is not in DET-10** and is clean with respect to RemoteCLIP.

Consequence: any retrieval-quality number measured on HRRSD using RemoteCLIP embeddings is inflated.
Preferred resolution — **build case memory from HRRSD, evaluate on NWPU VHR-10.** If the repo
currently does the reverse, flag it in the analysis. Either way, include a leakage-control ablation
using vanilla OpenCLIP ViT-B/32 (which has seen neither dataset); the RemoteCLIP-vs-OpenCLIP gap
being larger on HRRSD than NWPU quantifies the contamination.

### 4.2 Resolution and small objects

RemoteCLIP degrades outside its high-resolution training regime — its own paper reports zero-shot
accuracy on EuroSAT (64×64) falling 11.15 points below CLIP for ViT-B/32 and 24.83 for ResNet-50,
attributed to the resolution domain gap. Vehicle crops in NWPU VHR-10 are ~10–20 px before upsampling
to 224. **This is the highest-probability failure mode in the project** and is why §7 Step 2 is a
go/no-go gate rather than a task.

GSD differs across datasets: NWPU VHR-10 is 0.5–2 m for colour imagery; HRRSD is 0.15–1.2 m and mixes
Google Earth with Baidu Map sources. Define context-crop extent in **metres**, not pixels, or
normalise by GSD and document the choice.

### 4.3 Class taxonomy

NWPU VHR-10 has 10 classes; HRRSD has 13. The extra three (crossroad, T-junction, parking lot) are
**genuine open-set cases** — do not map them away. They are the novelty-detection test set: a correct
verification layer routes them to manual review via low max-similarity rather than confidently
accepting or rejecting. The label mapping must be an explicit, committed artifact, not inline dict
literals.

### 4.4 Split discipline

Images contributing cases to memory and images being evaluated must be **disjoint at the image ID
level**. Splits must be deterministic, seeded, and committed to disk as ID lists — not regenerated at
runtime.

### 4.5 FAISS is removed

Case memory tops out around 10⁴–10⁵ entries at 512-d. Brute-force `torch.matmul` over normalised
embeddings is milliseconds, and `IndexFlatIP` — the index that applies at this scale — *is* brute
force. Do not add FAISS. Document in limitations: *"exact search is O(N·d); beyond ~10⁶ cases an
IVF-PQ index would be required, which is an engineering swap, not a research question."*

---

## 5. Data models to establish (Step 0 of the work order)

```python
@dataclass
class Proposal:
    proposal_id: str          # f"{image_id}_{idx}"
    image_id: str
    dataset: str              # "nwpu" | "hrrsd"
    bbox: tuple[float, ...]   # xyxy, absolute px
    pred_class: str
    det_conf: float
    # assigned by GT matching:
    matched_gt_id: str | None
    iou: float
    gt_class: str | None
    verdict: str              # "accept" | "relabel" | "reject"
    obj_area_px: int          # for size-stratified analysis
```

**Detector must run at a low confidence floor (0.05–0.10), not 0.25.** The uncertain band is the
entire population the verification layer exists to triage. High-confidence-only proposals make the
task trivially easy and the coverage numbers meaningless. If the current predictions were generated
at 0.25, they must be regenerated.

**GT matching rule** (defines the labels — make it explicit and configurable):

- Greedy match each proposal to *unmatched* GT boxes by IoU, highest first. A GT box absorbs at most
  one proposal, otherwise post-NMS duplicates get labelled correct.
- `IoU ≥ 0.5` and `pred_class == gt_class` → **accept**
- `IoU ≥ 0.5` and `pred_class != gt_class` → **relabel** (target = `gt_class`)
- `IoU < 0.5` → **reject**

Run the main analysis at IoU 0.5; report a sensitivity row at 0.7 (Open Images V4 used 0.7 for
verification because a loose box is still a bad annotation).

**Case memory storage** — deliberately minimal:

- `embeddings.npy` — float32 `(N, 2, D)`, L2-normalised per stream, memmapped
- `cases.db` — SQLite, one row per case, `embedding_row_idx` as join key. Columns: `image_id`,
  `bbox`, `predicted_class`, `verified_class`, `detector_confidence`, `decision`, `provenance`,
  `timestamp`, `annotator_id`, `embedding_row_idx`.
- `timestamp` is mandatory — it is what makes the memory-growth streaming experiment possible.

---

## 6. Crop specification

- **Object crop**: expand bbox to square on the longer side, pad 10%, clip at image edge, resize to
  224. Square-before-resize avoids aspect distortion, to which CLIP is sensitive.
- **Context crop**: same centre, side = `k × max(w, h)`, `k` configurable (default 4). Near image
  edges, **shift the window inward rather than zero-padding** — black borders are a strong OOD signal
  that will dominate the embedding.
- **No minimum-size filter yet.** Record `obj_area_px`; filtering now hides the failure Step 2 exists
  to detect.
- Cache crops to disk as JPEGs keyed by `proposal_id`. Embeddings get recomputed several times during
  the bake-off; re-cropping each run wastes hours.

---

## 7. WORK ORDER

### Step 0 — Proposal records (foundation)
Regenerate predictions at low confidence floor. Implement GT matching. Emit `Proposal` records to
Parquet or JSONL. Commit deterministic split ID lists. **This artifact is what everything downstream
reads — get it right once.**

### Step 1 — Crops
Per §6. Cached, keyed, deterministic.

### Step 2 — Encoder bake-off ⚠️ **GO/NO-GO GATE**

Sample ~2,000 proposals stratified by class and size. Embed with four encoders. Compute **top-5
retrieval purity** — of the 5 nearest neighbours (excluding self, **excluding same source image**),
what fraction share the true class. Same-image exclusion is critical: adjacent detections in one tile
are near-duplicates and will inflate purity substantially.

| Encoder | <32² px | 32²–96² px | >96² px | All |
|---|---|---|---|---|
| RemoteCLIP ViT-B/32 | | | | |
| RemoteCLIP ViT-L/14 | | | | |
| OpenCLIP ViT-B/32 | | | | |
| DINOv2 ViT-B/14 | | | | |

Compare every cell against the **class prior** for that bucket. If `<32²` sits at the prior for all
encoders, retrieval carries no signal for small objects → scope the project's claims to `≥32²` and
state it explicitly. That is a legitimate scoping decision, not a failure, *provided it is found in
week 1*.

Note ViT-B/**32** patches a 224 input into only 7×7 tokens — coarse for small upsampled crops. ViT-L/14
will likely win; budget the compute. Load RemoteCLIP via `open_clip` + the released checkpoint state
dict (HF: `chendelong/RemoteCLIP`); follow the repo's "Load RemoteCLIP" section rather than guessing
filenames.

Also record from the same run: mean `cos(obj_emb, ctx_emb)` across the corpus. **If > ~0.9, the
two-stream design is one stream wearing a hat** and the masked-context variant is required before the
α-sweep means anything.

### Step 3 — Memory + retrieval
`s = α·(obj @ obj_mem.T) + (1−α)·(ctx @ ctx_mem.T)`, one matmul, top-k via `np.argpartition`.
Return **both positive and negative** neighbours — cases the annotator rejected are the reason this
beats same-class retrieval, and without them the VLM only ever sees confirming evidence.

Sweeps (all cheap, all are ablations you need anyway): α ∈ {0, 0.25, 0.5, 0.75, 1} — α=1 and α=0 are
your object-only and context-only ablations for free; context scale k ∈ {2, 4, 8}.

### Step 4 — Baselines and the gate
Feature vector: `[det_conf, sim@1, agreement@5, class_margin, log_area, novelty=1−sim@1]`.
Calibrate detector confidence first (temperature scaling on val) — raw YOLO scores are poorly
calibrated.

| # | Pipeline | Isolates |
|---|---|---|
| B0 | Calibrated detector confidence only | Does retrieval add anything? |
| B1 | + kNN label vote over embeddings | Is retrieval sufficient? |
| B2 | + logistic regression on retrieval features | Does *learning* beat voting? |
| B3 | + VLM on escalated band only | Does language reasoning add anything? |

**B2 is the number the VLM must beat.** Produce the B2 risk–coverage curve *before* writing a single
VLM prompt — otherwise prompts get tuned against a moving target with no reference. Add conformal
calibration so the auto-accept error rate is guaranteed ≤ α rather than threshold-tuned by eye.

### Step 5 — VLM (only after B2 exists)
- **Qwen2.5-VL-7B locally**, not GPT-4o: reproducibility (endpoints drift), cost, and generic VLMs
  are weak on overhead imagery and small objects. Keep one closed-model run as contrast if desired.
- **Invoked only on the escalated band.** Cuts cost ~5× and measures the VLM where it could plausibly
  help rather than diluting the effect across thousands of easy cases.
- Structured output via constrained decoding (outlines/xgrammar), not parse-and-retry — retry loops
  silently bias results toward cases the model finds easy to verbalise.
- Output schema must include `supporting_case_ids`. Measure **citation validity**: how often are the
  cited cases actually consistent with the verdict.
- Cache responses keyed on proposal hash.

### Step 6 — Ablations (the part that makes this research-flavoured)
- **Counterfactual evidence**: inject deliberately wrong retrieved cases. Always flips → VLM is a
  rubber stamp for retrieval. Never flips → retrieval is decorative. Quantify where you sit.
- **Evidence format**: retrieved cases as images vs. as text (labels, similarities, agreement counts).
  If text-only matches, the visual evidence channel is unused — a finding.
- **Label noise robustness**: inject {0, 5, 10, 20}% noise into memory, plot verification accuracy.
  Case memory is a positive feedback loop with no damping; this is the experiment that acknowledges it.
- **Memory growth** ⭐: process eval set as a stream, plot coverage@1%-risk vs `|memory|` at
  100/500/2k/10k. **This is the headline figure** — the only one showing something a static
  classifier cannot do. Compare against a **linear probe retrained on the same accumulated labels**;
  expect kNN to win early and the probe to overtake later. Reporting the crossover is a genuinely
  useful result.

### Step 7 — Minimal UI
Single Streamlit review-queue page. **Two days, week 7.** Not an artifact polished from week 3.

---

## 8. Reporting requirements

- Bootstrap confidence intervals **resampled at the image level, not proposal level**. Proposals
  within one image are strongly correlated; per-proposal bootstrapping produces fake-narrow intervals.
- Per-class breakdown, especially vehicle vs. large-area classes — aggregates are dominated by the
  most frequent class.
- Calibration: ECE + reliability diagram per pipeline.
- Asymmetric costs: accepting a wrong box corrupts the dataset permanently; rejecting a correct one
  costs one re-annotation. Weight accordingly and say so.
- mAP appears once, as a sanity-check table. Detector quality is not the contribution.

---

## 9. Framing (for the report and the final presentation)

*"We are not building a better detector. We are building a decision-support layer over any detector,
and the contribution is quantifying the effort/quality tradeoff it enables."*

Honest novelty audit — state this rather than overclaiming:
- **Standard engineering**: YOLO detection; CLIP-embedding kNN over a labelled cache (essentially
  Tip-Adapter, Zhang et al. ECCV 2022 — cite it, it legitimises the retrieval module); retrieval-
  augmented classification (RAC, Long et al. CVPR 2022); human verification as a cheap annotation
  primitive (Papadopoulos et al. CVPR 2016; production practice in Open Images V4, where ~10% of
  training boxes came from verification series); domain-specific CLIP (RemoteCLIP).
- **Mildly novel in composition**: dual object+context embeddings as retrieval evidence in overhead
  imagery; a growing verified-case memory as grounded evidence for a VLM verifier; the evaluation
  framing as selective prediction with a memory-growth curve.
- **Genuinely novel**: nothing — and that is fine for an 8-week course project.

If a negative result appears (VLM does not beat logistic regression), **report it as the finding**.
"Retrieval provides the signal; the language model adds cost and latency without adding discriminative
power" is a good result and a better interview story than a marginal win.

---

## 10. Timeline and freezes

| Week | Work |
|---|---|
| 1 | Steps 0–2. Encoder gate. |
| 2 | Detector calibration, proposals, crops finalised. |
| 3 | Memory + retrieval + sweeps. **ARCHITECTURE FREEZE at end of week.** |
| 4 | Baselines B0–B2, risk–coverage harness, conformal calibration. |
| 5 | VLM integration, structured output, citation validity. |
| 6 | Ablations (Step 6). **FEATURE FREEZE at end of week.** |
| 7 | Streamlit (2 days), figures, cost model. |
| 8 | Writing, buffer. |

Ranked risks: (1) small-object embeddings useless — mitigated by the week-1 gate; (2) engineering
consumes evaluation time — mitigated by the two freezes; (3) VLM adds nothing over logistic
regression — mitigated by pre-committing to report it; (4) domain shift destroys recall — check mAP
in week 2; (5) RemoteCLIP/HRRSD leakage — mitigated by the OpenCLIP control.

---

## 11. Working style

- Verb-led, dense, no filler. Uncomfortable assessments stated plainly.
- Concrete artifacts over frameworks. If a change can be made with a function, do not make it with a
  class hierarchy.
- Ask before assuming. If the repo is ambiguous, list the question — do not pick a default and
  proceed silently.
- Every threshold, seed, and path lives in config. No magic numbers in module bodies.
- When implementation and this brief disagree, surface the conflict rather than silently following
  either one.
