# STATE — resume pointer

**Rebuilt 2026-08-30** after an unclean unmount of the old working drive
(`/media/nishanksatish/Mark50`, NTFS via `ntfs3`) lost every write after
2026-08-12 22:53. The repo now lives at `~/retrieval-grounded-remote-sensing`.
Everything under `data/`, `reports/gate/`, `reports/phase1/`, `src/retrieval/`,
`scripts/` (except `run_b0.py`), and `configs/splits/` survived intact and is
the ground truth this file was rebuilt from — read the CSVs and reports
directly if a number below looks stale, they are the primary source, this
file is a summary of them.

`reports/FINDINGS.md` (the numbered [REAL]/[SYN]/[UNMEASURABLE] findings log)
and `reports/phase1/BASELINES.md` / `B3_VLM.md` were also lost and are being
rebuilt from the surviving CSVs — see "What's being rebuilt" below.

---

## Ownership

**I (Nishank) own `src/retrieval/` and the VLM half.** My teammate owns
`src/detection/` and `src/data/`. All my code goes in `src/retrieval/`;
I do not edit his directories — if something there blocks me, I ask him.

## Fixed decisions (do not re-litigate these)

- **Split**: `configs/splits/nwpu_vhr10_multilabel_split_seed42_fixed.csv`,
  562/120/118 train/val/test. **Key on `image_uid`, never `image_id`** — the
  export's `image_id` collides across positive/negative directories
  (`export_join.py` exists specifically to fix this).
- **Train + val, positive images only, for everything.** Test split is
  untouched throughout Phase 0/1.
- **No config framework, no plugin/registry system, no abstract
  `BaseEncoder` hierarchy, no FAISS, no database beyond SQLite for case
  memory, no CLI framework, no caching layer beyond plain `.npy` files.**
  Streamlit UI is explicitly authorised.
- **Same-image exclusion is not optional** — every retrieval query excludes
  all crops from its own source image (`gate.eligibility_mask`).
- Determinism: seed 42 everywhere, sorted listings, byte-identical reruns.
- **Vehicle numbers are RGB-only by default**, sensor-stratified always
  available, never quoted without saying which — 31/555 images (contiguous
  block 422–453) are Vaihingen CIR, and 70.7% of vehicle crops are CIR.
- **Standing rule: select every model/feature/hyperparameter on operating-
  point cost (coverage at a false-accept budget), never on AUROC.** Found
  in the α-sweep: AUROC picks α=1.0, which is 14× worse in false-reject
  cost than the α=0.5 the operating point picks. `phase1.selected_alpha()`
  is the one place this lives; downstream scripts read the winner back out
  of `alpha_sweep_summary.csv` rather than hardcoding it.
- **Never quote a combined false-accept/coverage number without the harm
  mix beside it** — the synthetic query mix is 53% box error, which is not
  realistic, and `mix_sensitivity.csv` exists because of this.
- **HRRSD numbers are never merged into NWPU numbers.**
- `NIM_API_KEY` is read from the environment only (`nim.api_key()`); `.env`
  is gitignored and never read by code.
- **Retraining/intermediate-checkpoint work is parked.** Class-error
  findings that depend on it stay tagged `[UNMEASURABLE]`, never
  `[REFUTED]`, until it resumes.
- Experimentation is closed: no new ablations, no more HRRSD runs, no more
  API calls, until further notice.
- Only commit to git when explicitly asked.

---

## What's proven (Phase 0 gate test — `reports/gate/GATE_REPORT.md`)

Scope: 555 positive images (train+val), 3,373 GT boxes, clean + jittered
(IoU 0.4–0.7) + background (999 crops) corpora.

- **Retrieval works.** Winning config `remoteclip_l14` / local stream:
  jitter macro P@5 **0.991** vs 0.100 chance. Survives detector-grade
  localisation error (0.005 P@5 cost at IoU 0.4–0.5).
- **No class fails**, including the detector's four worst classes
  (harbor 18% detector-correct → retrieval P@5 1.000; tennis_court 29% →
  0.992; basketball_court 37% → 0.965; ship 38% → 0.976).
- **Domain pretraining pays for itself**: RemoteCLIP B/32 beats OpenAI
  CLIP B/32 (same architecture) by +0.068 novelty AUROC and −9.1pp false-
  reject cost at 95% background rejection.
- **`novelty = 1 − sim@1` is a real gate feature** but only with L/14 local
  (AUROC 0.981, 8.8% false-reject cost at 95% bg rejection — every other
  encoder loses a third to two-thirds of real objects at the same budget).
  It detects "no object here", **not** "box is in the wrong place" — a
  jittered box only drops sim@1 by 0.015 vs clean, background drops it by
  0.078.
- **Both streams (local + regional) are kept** — paired cosine 0.761, well
  under the 0.90 redundancy line, and they fail in opposite directions on
  background source (negative-image AUROC vs positive-tile-empty-patch
  AUROC).
- **Open-set (leave-one-class-out, `reports/phase1/OPEN_SET.md`)**: novelty
  flags an unseen class with macro AUROC **0.999**, false-flag rate 0.2% at
  95% novel-detection — stronger than the background result, because an
  unseen class still embeds coherently, it just has no neighbour.

## What's proven (Phase 1 — α-sweep, baselines — all on synthetic proposals)

**Everything in this section is `[SYN]` — built from `src/retrieval/proposals.py`
on clean/jittered/background NWPU crops, not real detector output.** Memory
= 9,351 train-side cases (`data/memory/cases.db` + `embeddings.npy`), query
= 1,766 val-side proposals, 5 verdicts (`accept`, `relabel`, `adjust`,
`reject_localisation`, `reject_background`).

- **α = 0.5, k = 4** selected on `bg95_accept_cost` (`alpha_sweep_summary.csv`),
  unchanged across two proposal-set revisions.
- **B2 (logistic regression) beats B1 (kNN vote) beats random** on AURC
  (0.4249 / 0.4691 / 0.6979 — `baseline_summary.csv`), but the combined
  number is dominated by the unrealistic 53% box-error mix; per-harm is the
  readable view (`coverage_by_harm.csv`):
  - background_accept: B2 covers 90.1% of val at 1% false-accept budget.
  - class_error: B2 covers 71.7% at 1% budget.
  - localisation_accept / loose_box_accept: both baselines near-zero
    coverage — **retrieval genuinely cannot see box-tightness**, confirmed
    independently by Phase 0 §6.
- **`class_margin` bug found and fixed**: was computed inside the top-5
  window with a −1.0 sentinel, collapsing to a sign-flip on
  `agreement_at_k`. Now computed over the full positive memory pool
  (`fuse.class_margin`). Relabel AUROC still reads 1.000 after the fix —
  the *feature* is real, the *synthetic relabel task* is saturated
  (accept's minimum margin 0.0231 > relabel's maximum 0.0171, on clean
  pixels with only the label changed — do not quote this as evidence the
  gate catches real class errors).
- **Provenance damping** (`case_weights` in `fuse.features`, applied only
  to `agreement_at_k` and `class_margin`): at 20% label noise, damped
  class-reading features hold up much better than undamped
  (`label_noise.csv`: class_error coverage@1% falls from 0.7078→0.2918
  undamped vs 0.7220→0.5747 at auto_weight=0.25) while class-agnostic
  features (background, sim-based) barely move — confirming the design
  choice to damp only the class-reading half.
- **Arm ablation** (`arm_ablation.csv`): `local` alone gets 0.90/0.775
  coverage on background/class_error at 1%; adding `local_tight` and
  `regional` moves weight around (`arm_ablation_weights.csv`) but does not
  clearly beat single-arm `local` at the operating point — default arm set
  stays `("local", "local_tight")` (`phase1.ARMS`) pending B0.
- **Conformal calibration** (`conformal_thresholds.csv`,
  Clopper–Pearson upper bound, δ=0.1): only `background_accept` (α=0.01)
  and `class_error` (α=0.01, α=0.05) get usable, budget-respecting
  thresholds on the current synthetic mix; `combined` and
  `loose_box_accept` are degenerate (threshold=inf) because the harm
  never gets rare enough in this mix to calibrate against — expected, not
  a bug, given the 53% box-error mix.

## B0 — real detector proposals, measured 2026-08-30 (`reports/phase1/b0_vs_b2*.csv`)

**The headline result: on real detections, B2 does not beat raw detector
confidence, and it does not come close.** This is the first `[REAL]` result
in this project's gate evaluation, replacing the `[UNMEASURABLE]` tag that
sat here until today.

Export: the teammate's *old*, all-800-image run (`~/Desktop/detections.csv`,
no split column, no `inference_images.csv`) — joined to `image_uid` via
`export_join.uid_by_index()` over
`docs/shared_resources/datasets/raw/NWPU VHR-10 dataset` (re-discovering
`source_index`; verified against several rows by hand before trusting it),
then filtered to val. 791 detections landed on val images (189 of them on
background tiles — real false positives, not filtered out). Confidence
threshold was very low (min 0.050); 1,222/5,497 detections project-wide sit
below 0.25, so this export is closer to "everything the model considered"
than a deployment-ready output.

| baseline | AURC | cov@1% | cov@5% | AUROC (accept vs background) |
|---|---|---|---|---|
| B0 (raw detector confidence) | 0.1299 | 0.029 | 0.159 | **0.976** |
| B2 (logreg, fit on synthetic data) | 0.2550 | 0.004 | 0.004 | 0.788 |

- **B2 is not broken or anti-correlated.** Its individual features separate
  real accept from real background sensibly (sim@1 0.948 vs 0.893,
  agreement@k 1.000 vs 0.813 — both directionally identical to the
  synthetic/Phase-0 pattern, just a smaller gap), and its AUROC (0.788) is
  well above chance. It is simply **much weaker than B0's raw confidence**
  (0.976) on this export, where confidence itself is highly informative
  (accept mean confidence 0.872, reject_background mean 0.230).
- **Coverage@1%/5% (0.004) looks far worse than the AUROC gap (0.788 vs
  0.976) suggests**, because those budgets are about the very top of the
  ranking, and B2's probability calibration — fit entirely on the synthetic
  train corpus — evidently misorders a few real background detections into
  the top 1-5% of its ranking. This is a **calibration/transfer problem, not
  a feature problem**.
- **Suspect: `evidence_margin`, B2's second-largest weight (+1.96 on
  `local_tight`, +1.40 on `local`), barely separates on real data** (accept
  mean 0.003 vs reject_background mean −0.017 — almost no gap), while
  `agreement_at_k__local` (weight +2.61, the largest) separates far better
  (1.000 vs 0.813). B2 was fit on synthetic data where `evidence_margin`
  carried real signal; on this real export it looks close to noise, and the
  model still spends a lot of its decision boundary on it. Not confirmed as
  the sole cause — flagged as the leading hypothesis, not re-litigated
  further (no new experiments per the standing rule).
- **`class_error` harm is vacuous in this run: zero `relabel` verdicts**
  (`verdict_for`'s IoU/class-match rule never fired one on these 791
  detections — every box that overlapped a GT box at all had the matching
  predicted class). Its `cov@1%=1.000` in `b0_vs_b2_by_harm.csv` is a
  trivial artifact of `n_harmful=0`, not a real result — do not quote it.
- **B2+confidence variant was attempted and dropped**, not just skipped:
  it needs `detector_confidence` on the *training* holdout, which comes
  from `proposals.py`'s synthetic corpus and is `None` there by
  construction — casting to float produced NaN and `LogisticRegression`
  correctly refused to fit. Removed from `run_b0.py` rather than patched
  with an imputer (that would be inventing training signal, not measuring
  anything).

### Follow-up, same day: refit on real data, and the evidence_margin test

This export's `detections.csv` covers all 800 images (no split column), so
real *train*-split detections exist too — `build_real_proposals` was
generalised to pull either split, and `extract_detector_crops` was fixed to
key its output directory off each record's own `corpus` field rather than a
single hardcoded constant (it was writing every corpus to the same
directory, which silently mixed 3,998 train crops into the val corpus on
the first attempt — caught before it reached any reported number).

**B2 refit on 919 real train-split detections** (25% image-level holdout,
retrieved against the synthetic case memory with every case from a holdout
image excluded, so nothing retrieves its own tile) — applied unchanged to
the same 791 real val queries:

| baseline | AURC | cov@1% (background) | cov@5% (background) | AURC (background) |
|---|---|---|---|---|
| B0 | 0.1299 | 0.5512 | 0.6903 | — |
| B2 (synthetic-fit) | 0.2550 | 0.0038 | 0.0038 | 0.1532 |
| **B2, real-fit** | **0.1343** | **0.2857** | **0.6991** | **0.0570** |

**Confirms the fitting-artifact hypothesis, and more strongly than
recalled.** `background_accept` coverage@1% recovers **0.38% → 28.57%, a
75× jump** (the pre-loss run was recalled as "0.004 → 0.139", a 35×
recovery — this run recovers further; not an exact match to the specific
recalled figure, plausibly because this reconstruction's exact holdout
draw or export differs in some untraceable way from the original, but the
same conclusion, more strongly evidenced). AURC nearly matches B0 (0.0570
vs 0.0520). `localisation_accept` cov@1% goes 25.4% → 69.8%, past B0's
67.9%. **The *combined* cov@1%/5% barely move (0.38%→0.51%, 0.38%→24.9%)**
— per the standing rule never to quote combined without the mix, this is
the combined curve being bottlenecked by whichever single harm still has a
bad case near the very top of the ranking (only ~8 items sit inside a 1%
budget at n=791), not evidence the per-harm recovery isn't real.

**Evidence_margin hypothesis: tested, partially confirmed — a contributing
factor, not the sole cause.** Dropped `evidence_margin__local` and
`evidence_margin__local_tight`, refit on the *original* synthetic
fit_queries (not the real ones above), applied to the same val queries:

| baseline | AURC (background) | cov@1% (background) | cov@5% (background) |
|---|---|---|---|
| B2 (synthetic-fit, full) | 0.1532 | 0.0038 | 0.0038 |
| B2, no evidence_margin | 0.1123 | 0.0000 | 0.2592 |
| B0 (for reference) | 0.0520 | 0.5512 | 0.6903 |

AURC improves (0.1532→0.1123) and cov@5% recovers substantially
(0.4%→25.9%, ~68×) — real effect, evidence_margin genuinely does not
transfer and genuinely does drag the model down. But cov@1% does not
recover at all (nominally 0, though at n=791 that is a single item's rank
shift, not a robust zero), and AURC stays well short of both B0 and the
real-fit fix. **Verdict: evidence_margin is one identifiable, real
contributing cause, not the whole explanation** — the training
distribution shift is broader than this one feature, which is why the
full real-data refit (above) recovers much more than dropping this one
feature does.

**What this means for the project, stated plainly, per the standing "don't
hedge" rule**: the retrieval-based gate, as fit on synthetic data alone,
does not outperform thresholding the detector's own confidence on this
real export. But this is now demonstrated to be a **training-data
problem, not a features problem** — refit on real (even a modest 919-crop,
image-held-out) sample of real detections, B2 very nearly matches B0 on
AURC and substantially beats it on several per-harm coverage numbers.
`evidence_margin`'s poor transfer is a real, named, partial contributor,
not the whole story. The natural next step is a full real-data refit with
a larger, non-holdout-limited training sample, but that is a new
experiment and is not done here.

### Threshold recalibration, same day — hypothesis tested and refuted; real cause found

`app/gate_explorer.py`'s fixed `NOVELTY_ACCEPT_THRESHOLD = 0.9436` comes
from Phase 0's clean-GT-crop background distribution (`GATE_REPORT.md` §6,
`background_reject_rate=0.95`). `docs/UI_TEST_CASES.md` case B showed a
confident, correctly-localised real accept (`storage_tank`, sim@1 0.931)
escalating because it fell just under this threshold — the working
hypothesis was that real detector crops sit systematically lower in
similarity space than Phase 0's clean corpus, so a threshold recalibrated
on real data would be *more permissive* and recover coverage.

**Recomputed `gate.rejection_threshold` on real accept vs real
`reject_background` sim@1 (`sim_at_1__local`, n=486 accept / 189
background), at three operating points — the threshold moves the *wrong*
direction at every one of them:**

| background_reject_rate | threshold | real-accept coverage | real-bg leak |
|---|---|---|---|
| 0.90 | 0.9334 | 79.8% | 10.1% |
| **0.95** (Phase 0's own rate) | **0.9487** | **55.8%** | **5.3%** |
| 0.99 (this project's 1% budget) | 0.9601 | 27.4% | 1.1% |

**All three real-data thresholds are *higher* than 0.9436, not lower — the
hypothesis is refuted, robustly, not just at one operating point.** Fixing
the threshold to match Phase 0's own 5%-leak construction exactly
(0.9487 vs 0.9436) already costs coverage: 55.8% vs the old threshold's
63.4% at that same real accept population. Recalibrating alone makes case
B's problem *worse*, not better.

**Why: `sim_at_1` distributions.**

| | min | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| real accept | 0.856 | 0.950 | 0.969 | — | — | 0.992 |
| real `reject_background` | 0.781 | 0.900 | 0.933 | 0.949 | 0.960 | 0.969 |

Real accept and real background overlap far more than Phase 0's clean/
constructed corpora suggested — background's **upper tail** (p95=0.949,
p99=0.960) reaches almost into accept's own range, not because background
crops generally score higher, but because a threshold has to clear that
tail to hit a stated leak rate, and clearing it costs a lot of accept
coverage given how close the two distributions sit.

**Leading explanation, tied directly to case C**: `reject_background` here
is assigned by IoU-matching against NWPU's ground truth, which is not
guaranteed exhaustive. Case C (`docs/UI_TEST_CASES.md`) found several real
detections — `nwpu_positive_170`, `nwpu_positive_081_det005/006/008` —
sitting in `reject_background` (`iou=0.0`) while scoring sim@1 0.948–0.952,
essentially inside the accept distribution, because they are real objects
NWPU's GT never boxed. Those are exactly the kind of case that inflates
`reject_background`'s upper tail. **This is a different, better-evidenced
mechanism than "distribution shift"** — not a wrong calibration, but a
contaminated label: some fraction of what's counted as `reject_background`
is not background at all. Not chased further (would mean manually
auditing which `reject_background` cases are true negatives vs annotation
gaps — a real, larger piece of work, not done here).

**Whether this also explains B2's collapse (findings 19–24) is not
established.** `evidence_margin`'s poor transfer (finding 24) was tested
independently on the same 791 val proposals; both mechanisms are real, but
nothing here shows they're the same one — B2's degradation may partly
share this same contaminated-`reject_background` root cause (its
`sim_at_1`/`agreement_at_k` features would inherit the same overlap), but
that is stated as a plausible connection, not verified.

**App update**: `NOVELTY_ACCEPT_THRESHOLD` was changed to **0.9601** (the
1%-budget value, matching this project's standard false-accept convention
used everywhere else) with the 0.9487/5%-leak alternative noted in a
comment. Consequence, stated plainly: **the app will now escalate *more*
real accepts than before, not fewer** — this is the honest number, not a
fix for case B's complaint. `docs/UI_TEST_CASES.md` case B is corrected
accordingly; cases C and D both flipped from `accept` to `escalate` under
the new threshold (re-walked and confirmed live, `docs/UI_TEST_CASES.md`).

## Near misses — plausible output, no error raised

Bugs and near-bugs where the code ran cleanly, produced a plausible-looking
number, and the number was wrong. Kept as one list because they're the same
failure class and worth scanning together before trusting a new measurement.

- **`class_margin` sentinel collapse** (finding 12 above): computed inside
  the top-5 window with a −1.0 sentinel, took exactly two values, and read
  as a real feature until the range was inspected.
- **Single-branch logprob renormalisation** (`yes_probability` docstring,
  `scripts/run_b3_vlm.py`): an endpoint that returns only the argmax
  logprob would, if renormalised over {yes, no} like a full top-20
  response, hand back a constant 1.0 for every call — a "working" feature
  that is in fact a constant. Handled by using the absolute probability
  instead when only one branch is visible; caught before it shipped, not
  after.
- **Reasoning-model forward-scan false positive** (2026-08-31, this
  session): fixing `yes_probability` for a reasoning model by scanning
  forward for the first yes/no-shaped token looked like the obvious fix
  and ran without error — it matched the word "yes" *inside the literal
  phrase "a yes/no question"* in the model's own reasoning trace, on 3 of
  5 sample crops, silently reading logprobs at the wrong position. Caught
  only by manually inspecting the token sequence, not by any exception.
  Fixed by scanning from the end instead (`find_answer_position`,
  `scripts/run_b3_vlm.py`).
- **QuickGELU** — gap filled 2026-09-01, found directly in the existing
  code rather than reconstructed from memory: `open_clip` only *warns*
  when a model's activation function config doesn't match its pretrained
  weights' tag (e.g. building plain `ViT-B-32` and loading OpenAI's
  `openai`-tagged weights, which need the `-quickgelu` variant), then
  silently builds a model whose activation disagrees with its own weights
  anyway. For the OpenAI CLIP control specifically, that would have
  quietly handicapped the very baseline RemoteCLIP is measured against —
  every RemoteCLIP-vs-OpenAI-CLIP number in `GATE_REPORT.md` would rest on
  a broken control. `encoders.py`'s `_load_open_clip` catches this warning
  and raises instead (`if mismatch: raise RuntimeError(...)`).
- **A randomly-initialised RemoteCLIP encoder would look like a working
  one** (2026-09-01, prompted by the user noticing
  `WARNING:root:No pretrained weights loaded for model 'ViT-L-14'. Model
  initialized randomly.` in every session log this whole time). Investigated
  fully rather than assumed either way: the warning is real but benign —
  it fires from `open_clip.create_model_and_transforms(..., pretrained=None)`
  during RemoteCLIP's two-step load (empty architecture first, then its
  real weights via an explicit `hf_hub_download` + `load_state_dict`
  immediately after) and is emitted via `logging.warning()` to the root
  logger, not `warnings.warn()` — so the QuickGELU check above, which only
  watches `warnings`, never saw it. Verified directly, not inferred: the
  resolved checkpoint path, `load_state_dict` reporting 0 missing/0
  unexpected (302 tensors on ViT-B-32, 446 on ViT-L-14 — the user's
  recalled "302" was the B/32 check specifically), the model's actual
  parameter values matching the checkpoint's exactly and differing from an
  independent random-init model, RemoteCLIP's weights differing from
  OpenAI CLIP's, and every `detector`/`detector_train` embedding computed
  this session matching a fresh, forced recomputation to floating-point
  precision (cosine similarity 1.0) — nothing downstream is contaminated.
  **Still fixed anyway**, because "this specific instance turned out fine"
  is not the same guarantee as "this can't fail silently": the benign
  warning is now suppressed (so it can't be mistaken for the real thing
  again), and checkpoint fetch/read failures are now wrapped so any error
  surfaces as an explicit `RuntimeError` naming RemoteCLIP, never a silent
  fall-through to the empty architecture. `src/retrieval/encoders.py`.
- **Retry logic that didn't distinguish failure classes** (2026-08-31, same
  session, hours later): the first patient-retry design caught every
  `RuntimeError` identically and retried all of them with the same
  30s→5min backoff, up to 40 times. Two crops hit a genuinely
  unretryable failure — "no yes/no token anywhere in the response" — which
  is deterministic at `temperature=0.0` and will reproduce every time; the
  retry loop burned **~3 hours and ~35 wasted (uncached) API calls** on
  those two crops alone before this was noticed, and was still running
  when caught. Different flavour from the others above (an exception *was*
  raised and *was* handled, just not differently enough) but the same
  underlying lesson: a broad `except RuntimeError` silently conflates
  "worth retrying" with "will never succeed", and the code ran for hours
  without producing an error that would have flagged the distinction.
  Fixed with a separate `UnscorableResponse` exception class (one quick
  retry, then give up) versus the plain `RuntimeError` transport-failure
  path (the full patient schedule) — `scripts/run_b3_vlm.py`.

## B3 — VLM escalation-tier reader (`reports/phase1/b3_vlm_*.csv`)

Scope: only the escalated tail of B2's ranking, only the class-error
decision, prompt = "does this crop show a `{predicted_class}`?", `p(yes)`
via logprobs (`run_b3_vlm.py:score_logprob`).

**2026-08-26**: `nvidia/nemotron-nano-12b-v2-vl` retired by NVIDIA (HTTP
410). **2026-08-31/09-01**: rescored all three evidence arms on
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (a reasoning model — see
"Near misses" above for the scoring-path fix this required, and the
retry-logic bug caught mid-run). 2/516 crops in each arm are permanently
unscorable (the model never emits yes/no; deterministic at
`temperature=0.0`, so not a transient failure) and are excluded from that
arm's numbers.

| evidence | old model (retired) | new model (reasoning) |
|---|---|---|
| none | 0.744 | 0.759 (AURC 0.0092) |
| **true** | 0.796 | **0.845** (AURC 0.0084) |
| counterfactual | 0.743 | 0.725 (AURC 0.0106, **first real measurement** — the old run's counterfactual CSV never survived) |

**Every finding here reproduces, and the evidence-grounding one reproduces
more cleanly than before**: true evidence beats both none and
counterfactual by a wider margin on the new model (true−none = 0.086,
true−counterfactual = 0.120) than the old model showed (≈0.05 range,
counterfactual previously unmeasured). Coverage@1% for class_error goes
B2's 0.7078 → 0.7588 (none) → 0.8448 (true) either way. `evidence=none`
and `evidence=counterfactual` sit close together (0.759 vs 0.725) as
expected for a working control.

Historical, not rescored: `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`
(evidence=none): 0.7078 → 0.7559, weaker than the old 12B model's
contemporary result.

---

## What's been rebuilt (this session, 2026-08-30)

1. `docs/STATE.md` — this file. Done, including the B0 real-data result above.
2. `scripts/run_b0.py` — real-detector-proposal pipeline. **Done, ran
   successfully.** Detections joined via `export_join.uid_by_index()`
   (source_index route — this export has no `inference_images.csv`),
   filtered to val, scored B0 vs B2 per harm. See the B0 section above.
3. `reports/FINDINGS.md` — full [REAL]/[SYN]/[UNMEASURABLE]-tagged findings
   log. The numbered list (was findings 1–40) is not reconstructable
   verbatim from memory; rebuilt from the CSVs/reports, so the numbering
   will not match any previously-cited finding number exactly — flag this
   to the teammate if he has old citations to specific numbers. Being
   updated now with the B0 result as a new finding.
4. `app/gate_explorer.py` — Streamlit demo, six linear steps, rebuilt from
   this conversation's own transcript, then adapted to load real detections
   via the same source_index join as `run_b0.py`. Not yet walked through
   live in a browser.
5. `docs/UI_TEST_CASES.md` — written as prescribed walkthroughs before the
   app had real data; still needs an actual pass now that `detections.csv`
   exists.

## Open items

- **B2 needs to be refit on real (not synthetic) data before it is a
  credible gate.** The B0 result above is the reason: synthetic-fit B2
  loses to raw detector confidence on real proposals. A real *train*-split
  export (not just val) would be needed to refit it properly — this is a
  new experiment, not started.
- Pending, not yet started: rescale the similarity display in
  `app/gate_explorer.py` to a 0–100 legible scale (val-derived anchors:
  mean top-1 sim for true accepts vs background false positives), display
  only, raw cosine shown alongside, anchors in config with a staleness
  comment.
- Live walkthrough of `app/gate_explorer.py`'s six steps with real data —
  not yet done as of this writing.
