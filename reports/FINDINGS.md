# Findings

**Rebuilt 2026-08-30** from the surviving CSVs and reports after a data-loss
event (see `docs/STATE.md`). The original numbered findings log (which ran
1–40) did not survive; this is a fresh numbering with the same substance,
re-derived by reading the underlying report files and CSVs directly rather
than from memory. **Do not cite a number here as matching an earlier
conversation's citation of the same number** — the content should still be
correct, the numbering is not guaranteed to line up.

## Tag legend

- **`[REAL]`** — measured on real imagery / real ground truth, and the
  measurement does not depend on the synthetic proposal construction in
  `src/retrieval/proposals.py`. Phase 0's purity, novelty and open-set
  numbers are this: real crops, real classes, real background tiles.
- **`[SYN]`** — measured on the synthetic proposal set (`proposals.py`):
  clean GT = `accept`, jittered boxes = `relabel`/`adjust`, `jitter_low` =
  `reject_localisation`, constructed background crops = `reject_background`.
  Everything in Phase 1 (α-sweep, B1/B2 baselines, conformal, B3 VLM) is
  `[SYN]` until real detector proposals (`scripts/run_b0.py`) replace the
  query side. **The harm mix in these numbers (53% box error) is a
  construction artifact, not a measured detector error rate — never quote
  a combined risk/coverage number from a `[SYN]` finding without saying so.**
- **`[UNMEASURABLE]`** — cannot be measured in the current state, most often
  because it depends on the parked retraining work or on `scripts/run_b0.py`
  actually running (blocked on `detections.csv`, see `docs/STATE.md`).
  Never promoted to `[REFUTED]` — absence of a real-detector measurement is
  not evidence against the synthetic one, it is just a gap.

---

## Retrieval mechanics — `[REAL]`, source: `reports/gate/GATE_REPORT.md`

1. **Retrieval separates class at P@5 0.991 under detector-grade
   localisation error** (`remoteclip_l14`, local stream, jitter corpus,
   3,373 boxes vs 0.100 chance). Clean-GT P@5 is 0.996 — the two are close
   because class survives even a half-off box for these ten visually
   distinct classes.
2. **No NWPU class fails**, and retrieval is strongest exactly where the
   detector is weakest (harbor: 18% detector-correct, 1.000 retrieval P@5;
   tennis_court: 29% → 0.992; basketball_court: 37% → 0.965; ship: 38% →
   0.976).
3. **Domain pretraining is not a rounding difference — it decides whether
   novelty works at all.** RemoteCLIP B/32 vs OpenAI CLIP B/32 (identical
   architecture): +0.018 clean P@5, +0.068 novelty AUROC, −9.1pp false-
   reject cost at 95% background rejection. DINOv2 loses to both on
   novelty (AUROC 0.791).
4. **`novelty = 1 − sim@1` is a real gate feature, encoder-specific.**
   `remoteclip_l14`/local: AUROC 0.981, 8.8% real objects lost at 95%
   background rejection. Every other encoder in the comparison costs 33%
   to 63% at the same budget — encoder choice is load-bearing for this
   feature, not a preference.
5. **Novelty detects "no object here", not "box is in the wrong place".**
   Jittered box (IoU 0.4–0.5) drops sim@1 by only 0.015 from clean;
   background drops it by 0.078 — five times further. A separate
   localisation-quality signal is needed and cannot come from retrieval
   similarity.
6. **Both streams (local, regional) are kept — not redundant.** Paired
   cosine 0.761 (l14), well under the 0.90 line, and they fail in opposite
   directions on background source: local is better on object-free patches
   *inside* a positive tile (AUROC 0.978 vs regional's 0.898), regional is
   better on whole-tile negatives (0.997 vs 0.984).
7. **The local crop is not a pure object view.** Mean box fill is below 0.6
   for 7 of 10 classes (harbor 0.468, ship 0.505, bridge 0.553) — the
   streams are named for the scale they cover, not for what they contain,
   because for elongated classes "local" is already half surroundings.
8. **A hidden CIR/RGB sensor split exists (31/555 images) but is not doing
   vehicle retrieval's work.** RGB-only vehicle queries score P@5 0.993,
   indistinguishable from CIR (0.997) — the confound was real (70.7% of
   vehicle crops are CIR) but checked and ruled out. Vehicle numbers are
   reported RGB-only by convention (`purity_by_sensor.csv` carries the
   stratified version).
9. **Open-set: novelty flags an unseen class, decisively, for all ten
   classes** (`reports/phase1/OPEN_SET.md`, leave-one-class-out). Macro
   AUROC 0.999, macro false-flag rate 0.2% at 95% novel-detection, holding
   even for classes with as few as 16–19 held-out queries. Stronger than
   the background-rejection result, because an unseen class still embeds
   coherently — it just has no neighbour. Not yet tested with the
   localisation error a real detector would add (compounding effect
   unmeasured), and HRRSD's three open-set classes (`crossroad`,
   `parking_lot`, `t_junction`) are a harder, unrun complement.

## Gate baselines — `[SYN]`, source: `reports/phase1/ALPHA_SWEEP.md`, CSVs

10. **Selecting on AUROC instead of operating-point cost would have thrown
    away the regional stream and taken a 14× worse false-reject cost** —
    the finding that produced the project's standing selection rule. α=1.0
    wins macro AUROC (0.890); α=0.5 wins the actual deployed metric
    (0.6% vs 8.1% cost at 95% background rejection). `k=4` over `k=2`
    reproduces the same pattern independently (4.4× cost difference for
    +0.008 AUROC in AUROC's favour).
11. **B2 (logistic regression) beats B1 (kNN vote) beats random** on AURC
    (0.4249 / 0.4691 / 0.6979), but the *combined* number is dominated by
    an unrealistic 53% box-error mix — read per-harm instead
    (`coverage_by_harm.csv`): B2 covers 90.1% of val at a 1% false-accept
    budget for background, 71.7% for class-error, and near-zero for
    localisation — retrieval genuinely cannot see box tightness, which is
    the same conclusion as gate-mechanics finding 5, reached independently
    from proposal-level data.
12. **`class_margin` was broken (sentinel collapse) and is now fixed.**
    Old version took two values (±(1+sim@1)) because a −1.0 sentinel fired
    on every relabel query at near-1.0 purity; it was a sign-flip on
    `agreement_at_k` wearing a margin's name. Fixed version, computed over
    the full positive memory pool, takes 1,011 distinct values on 1,015
    queries. Relabel AUROC still reads 1.000 after the fix — **this is the
    synthetic relabel task being saturated, not evidence the feature
    generalises to real class errors.** Accept's minimum margin (0.0231)
    barely clears relabel's maximum (0.0171): a real but razor-thin gap on
    a task built from clean pixels with only the label swapped.
13. **Provenance damping protects the class-reading features under label
    noise, and leaves the class-agnostic ones alone by design.** At 20%
    corrupted memory labels, undamped class_error coverage@1% collapses
    0.7078→0.2918; damping at weight 0.25 holds it at 0.7220→0.5747.
    Background coverage moves only −3.6pp across the same noise range,
    confirming the deliberate choice to damp only `agreement_at_k` and
    `class_margin`, not `sim_at_1`/`novelty`/`evidence_margin`.
14. **Arm ablation: single-arm `local` is competitive with the two-/three-
    arm defaults at the operating point** (`arm_ablation.csv`) — 0.8997/
    0.7751 coverage on background/class_error at 1%, close to or better
    than adding `local_tight` and `regional`. The default feature set
    (`phase1.ARMS = ("local", "local_tight")`) is unchanged pending real
    detector proposals, deliberately — this is exactly the kind of choice
    that should be re-checked once B0 data exists.
15. **Conformal calibration (Clopper–Pearson, δ=0.1) only produces usable
    thresholds for `background_accept` and `class_error`** on the current
    mix (`conformal_thresholds.csv`); `combined` and `loose_box_accept`
    are degenerate (threshold = ∞) because those harms never get rare
    enough in a 53%-box-error synthetic mix to calibrate against. Expected
    given finding 11, not a calibration bug.

## VLM escalation tier (B3) — `[SYN]`, source: `reports/phase1/b3_vlm_*.csv`

**Model retirement, 2026-08-26 → rescored 2026-08-31/09-01.**
`nvidia/nemotron-nano-12b-v2-vl` (findings 16–18's original model) was
retired by NVIDIA mid-project (HTTP 410, "reached its end of life").
Replaced with `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. **Two
comparability caveats on everything below**: (1) it is a different model,
obviously; (2) it is a *reasoning* model, which changed the scoring
mechanism itself — the old `yes_probability(entries[0])` assumed the first
generated token was the answer, which no longer holds (the model opens
every response with a `<think>` block). Fixed by scanning the generated
sequence for the last yes/no-shaped token instead
(`find_answer_position`, `scripts/run_b3_vlm.py`) — validated on 20 real
crops (20/20 readable) before trusting it on the full band. See
`docs/STATE.md`'s near-misses section for two bugs caught building this:
a forward-scan false positive (matched "yes" inside the reasoning
phrase "a yes/no question"), and a retry-logic bug that burned ~3 hours
on two crops whose failure was deterministic, not transient.

16. **The VLM improves coverage on B2's escalated tail — reproduced with
    the new model.** Old (`nemotron-nano-12b-v2-vl`), evidence=none:
    class-error coverage@1% 0.7078 → 0.7441. New
    (`nemotron-3-nano-omni-30b-a3b-reasoning`), evidence=none: 0.7078 →
    **0.7588** (AURC 0.0092). Same direction, similar magnitude, different
    model and scoring mechanism — the finding survives the swap.
17. **Evidence-grounding is real, and the new model shows it more
    cleanly than the old one did.** Evidence=true: old model reached
    0.7961; new model reaches **0.8448** (AURC 0.0084) — the largest
    coverage number in this whole comparison. **The counterfactual
    control, missing from the original run, now exists**: evidence=
    counterfactual scores **0.7248** (AURC 0.0106) — close to the
    evidence=none baseline (0.7588) and far below evidence=true (0.8448).
    This is exactly the pattern a genuine evidence effect predicts: real
    evidence helps, fake evidence (same format, wrong labels) does not.
    The true-vs-counterfactual gap is **wider** with the new model (0.120
    vs the old model's inferred ~0.05 range) — the finding is not just
    reproduced, it is more convincing than before.

    | evidence | old model (retired) | new model (reasoning) |
    |---|---|---|
    | none | 0.744 | 0.759 |
    | **true** | 0.796 | **0.845** |
    | counterfactual | 0.743 | 0.725 |

18. **The smaller old model got a smaller gain** (historical, not
    rescored): `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`, evidence=none:
    0.7078 → 0.7559, weaker than the 12B model's contemporary result. Not
    re-run against the new model; not the comparison this session's rescore
    was for.

## Real detector proposals (B0) — `[REAL]`, source: `reports/phase1/b0_vs_b2*.csv`, measured 2026-08-30

19. **On real detections, B2 does not beat raw detector confidence, and not
    narrowly.** B0 (raw confidence) scores AUROC 0.976 (accept vs
    background), AURC 0.1299, coverage@1%/5% 2.9%/15.9%. B2 (the synthetic-
    fit logreg gate) scores AUROC 0.788, AURC 0.2550, coverage@1%/5%
    0.4%/0.4%. This is the first `[REAL]` measurement of the gate's actual
    deployment performance, resolving what was `[UNMEASURABLE]` in every
    prior finding above. **Stated plainly, per the standing no-hedging
    rule: as currently fit, the retrieval gate is not a credible
    replacement for thresholding the detector's own confidence.**
20. **B2's real-data AUROC (0.788) is genuine signal, not a broken
    pipeline** — its individual features separate real accept from real
    background in the expected direction (sim@1 0.948 vs 0.893,
    agreement@k 1.000 vs 0.813, both qualitatively matching the Phase 0/
    synthetic pattern, just with a smaller gap). The gap between that
    AUROC and B0's is a **generalisation/calibration problem**: B2 was fit
    entirely on `proposals.py`'s synthetic corpus and evidently misranks a
    handful of real background detections into the top of its ranking,
    which the strict 1%/5% budgets punish far more than AUROC does.
    Confirmed by findings 23–24 below: refitting on real data closes most
    of the gap, and `evidence_margin` (the leading suspect feature) is a
    real but partial contributor.
21. **`class_error` is vacuous in this measurement** — zero of 791 real
    detections landed a `relabel` verdict (every box overlapping a GT box
    had the matching predicted class), so the `class_error` row in
    `b0_vs_b2_by_harm.csv` (coverage@1% = 1.000 for both baselines) is a
    trivial artifact of `n_harmful = 0`. Do not cite it as "the gate
    catches 100% of real class errors" — there were none in this sample to
    catch.
22. **A B2+confidence variant (retrieval features plus detector
    confidence as a 12th column) cannot be fit from available data.** The
    training holdout is drawn from the synthetic corpus, which sets
    `detector_confidence = None` by construction — attempted, produced a
    NaN feature column, and was removed rather than patched with an
    imputer.
23. **Refitting B2 on real train-split detections recovers most of the
    gap — this is a training-data problem, not a features problem.**
    Detections.csv turned out to cover all 800 images (no split column),
    so 919 real train-side detections existed to refit on (25%
    image-level holdout; the synthetic memory drops every case from a
    holdout image so nothing retrieves its own tile). Applied to the same
    791 val queries: `background_accept` coverage@1% **0.38% → 28.57%, a
    75× recovery** (AURC 0.1532 → 0.0570, matching B0's 0.0520);
    `localisation_accept` coverage@1% 25.4% → 69.8%, past B0's 67.9%. The
    *combined* coverage barely moves (0.38%→0.51%) — consistent with
    finding 11's standing warning never to read combined without the
    per-harm breakdown, since at n=791 a 1% budget is ~8 items and one
    bad admission of any type dominates it. **Caught and fixed en route**:
    `extract_detector_crops` hardcoded its output directory to a single
    corpus constant, silently writing 3,998 real train crops into the
    val-only corpus directory on the first attempt — caught before it
    reached a reported number, fixed to key the directory off each
    record's own corpus.
24. **The `evidence_margin` hypothesis (finding 20) is tested and
    partially confirmed: a real contributing cause, not the sole one.**
    Dropped `evidence_margin__local`/`__local_tight` from the *original*
    synthetic fit (not the real-data refit above) and refit: AURC improved
    0.1532→0.1123 and `background_accept` coverage@5% recovered 0.4%→25.9%
    (~68×) — a real, measurable effect. But coverage@1% did not recover at
    all (stayed at ~0), and AURC remained well short of both B0 (0.0520)
    and the full real-data refit (0.0570 — finding 23). **Verdict:
    evidence_margin genuinely does not transfer from synthetic to real
    data and genuinely drags the synthetic-fit model down, but it is one
    identifiable piece of a broader training-distribution shift, not the
    whole explanation.**

## Still not measured

25. **`[UNMEASURABLE]`** — everything else in this document above the
    synthetic-proposal line, on real data: the α-sweep's selected α/k, the
    class-margin fix's real-data behaviour, provenance damping, conformal
    thresholds, and all of B3, none of which have been re-run against real
    detections. Only B0-vs-B2 and its two follow-ups (findings 19–24) have
    been measured for real so far.
26. **`[UNMEASURABLE]`** — class-error findings that depend on the parked
    retraining work. Stay tagged this way, never `[REFUTED]`, until
    retraining resumes.
27. **`[UNMEASURABLE]`** — localisation error compounding with open-set
    novelty (OPEN_SET.md limitation 2), and the HRRSD open-set complement
    (`crossroad`/`parking_lot`/`t_junction`, limitation 3).
