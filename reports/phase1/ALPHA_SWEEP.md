# Phase 1 — the α-sweep, and how comparisons are decided

Encoder `remoteclip_l14`, regional scale k=4 for the headline, seed 42.
Memory = train proposals, queries = val proposals, **test untouched**.
Both sides are synthetic proposals (`src/retrieval/proposals.py`) — no detector export yet.

---

## Methodology note: select on operating-point cost, never on AUROC

**This is a finding, not a hyperparameter preference, and it applies to every comparison in
this project from here on — including B0–B3.**

AUROC integrates over every possible threshold. A deployed gate runs at exactly one. When a
score's *ranking* is good on average but its *tail* is bad, AUROC stays high while the
operating point collapses — and the operating point is what decides how many correct
proposals get needlessly escalated to a human.

The α-sweep is the case that forced the rule. Ranked by macro AUROC the answer is α = 1.0,
i.e. discard the regional stream entirely. Ranked by the cost of running at 95% background
rejection, α = 1.0 is the **worst non-degenerate setting on the board**:

| α | macro AUROC | false-reject cost at 95% background rejection |
|---|---|---|
| 0.50 | 0.881 | **0.6%** |
| 0.75 | 0.889 | 0.9% |
| 1.00 | **0.890** (best) | 8.1% — 14× worse |

Selecting on AUROC would have thrown away the regional stream and taken a 14× increase in
wrongly escalated proposals for +0.009 of an integral nobody operates at.

**Rule adopted:** every model, feature and hyperparameter choice is decided on the cost at
the stated operating point. AUROC may be *reported* as a summary — it is comparable across
datasets in a way that a threshold is not — but it never selects. `scripts/run_alpha_sweep.py`
implements this in `min(..., key=bg95_accept_cost)`, and downstream scripts read the winner
back out of `alpha_sweep_summary.csv` via `phase1.selected_alpha()` so the rule lives in one
place.

Mechanism, for the write-up: the regional stream does not improve average separation. It
tightens the **lower tail of the accept distribution** — the handful of accepts whose local
crop resembles nothing in memory but whose surroundings are unremarkable. That tail is
exactly where a 95%-rejection threshold sits, and it is invisible to an integral over all
thresholds.

### α did not move when the sweep was put on the shared proposal set

Recorded because a hyperparameter that shifts silently looks like cherry-picking later.

`run_alpha_sweep.py` used to build its own corpora, so it was running on a **different
proposal set from the baselines** — its copy predated the `adjust` verdict and the
`jitter_low` corpus, and would have kept sweeping α on four verdicts while the baselines
scored five. It was collapsed onto `phase1.prepare()`, the same setup every other script
uses.

| | before (own corpora, 4 verdicts) | after (shared setup, 5 verdicts) |
|---|---|---|
| memory | 5115 cases | 9351 cases |
| query | 1015 proposals | 1766 proposals |
| **selected α** | **0.5** | **0.5** — unchanged |
| bg95 cost at α=0.5 | 0.0056 | 0.0056 |
| macro AUROC at α=0.5 | 0.881 | 0.868 |
| selected k | 4 | 4 — unchanged |

**α and k are unchanged.** The macro AUROC moved (0.881 → 0.868) because the verdict mix
changed underneath it — `reject_localisation` is now the IoU<0.4 corpus and `adjust` is a
new column — but the operating-point criterion picked the same α, which is the outcome the
standing rule is supposed to produce: a criterion tied to a fixed threshold is less
sensitive to mix than an integral over all thresholds.

The earlier 0.75 → 0.5 shift has a different cause and is recorded separately: that one came
from hardening the relabels and fixing `class_margin`, not from a code path change.

---

## Proposal set (hardened, see §4)

| verdict | source | memory (train) | query (val) |
|---|---|---|---|
| accept | clean GT | 2842 | 531 |
| relabel | jitter IoU ≥ 0.5, class swapped | 570 | 128 |
| adjust | jitter IoU 0.4–0.7, class right | 2272 | 403 |
| reject_localisation | jitter_low IoU < 0.4 | 2842 | 530 |
| reject_background | background crops | 825 | 174 |
| **total** | | **9351** | **1766** |

Memory: `data/memory/cases.db` + `embeddings.npy`, (9351, 3, 768), streams
`['local', 'local_tight', 'regional_k4']`, with a `dataset` column. No image_uid appears in
both memory and query (asserted).

**The mix is now 53% box error** (adjust + reject_localisation), which is *less* realistic
than the 17.9% it replaced — adding `jitter_low` gave the redraw case a population, but it
also doubled the box-error mass. The combined risk curve is correspondingly meaningless and
the per-harm columns in BASELINES.md are the only readable view. Recorded rather than tuned:
the mix is a construction artifact either way, and the fix is real detector proposals.

---

## The sweep at k=4

`P@5` is agreement@5 on accept queries. AUROC columns separate accept from each reject type
on that type's target feature. **`bg95 cost` is the selection criterion.**

| arm | α | P@5 | AUROC bg | AUROC loc | AUROC relabel | macro | **bg95 cost** |
|---|---|---|---|---|---|---|---|
| local | 0.00 | 0.953 | 0.956 | 0.553 | 0.997 | 0.764 | 38.4% |
| local | 0.25 | 0.999 | 0.994 | 0.760 | 1.000 | 0.838 | 3.0% |
| **local** | **0.50** | **1.000** | **0.998** | 0.848 | 1.000 | 0.868 | **0.6%** ← selected |
| local | 0.75 | 1.000 | 0.998 | 0.882 | 1.000 | 0.882 | 0.9% |
| local | 1.00 | 0.999 | 0.981 | 0.892 | 1.000 | 0.885 | 8.1% |
| local_tight | 0.00 | 0.953 | 0.956 | 0.553 | 0.997 | 0.764 | 38.4% |
| local_tight | 0.25 | 0.999 | 0.992 | 0.759 | 1.000 | 0.843 | 3.8% |
| local_tight | 0.50 | 0.999 | 0.996 | 0.854 | 1.000 | 0.880 | 1.5% |
| local_tight | 0.75 | 0.999 | 0.996 | 0.896 | 1.000 | 0.900 | 1.3% |
| local_tight | 1.00 | 0.996 | 0.965 | 0.902 | 1.000 | 0.898 | 10.5% |

Both arms show the same shape: an interior optimum on cost, a monotone climb on AUROC, and
the two disagreeing. The regional stream remains blind to localisation at every α
(0.892 → 0.553 as α goes 1.0 → 0.0), and `local_tight` remains the better localisation
detector at every α while `local` is the better background detector.

**The localisation AUROC column jumped from ~0.65 to ~0.85 versus the previous run, and that
is a definition change, not an improvement.** `reject_localisation` now means IoU < 0.4
rather than IoU < 0.5, and boxes that badly broken *are* detectable. The 0.4–0.7 boxes that
retrieval genuinely cannot see moved to `adjust`, where they are measured separately. Do not
compare this column against the earlier one.

### k sensitivity, at the selected α = 0.5 (local arm)

Run only at the selected α, not as a third sweep axis.

| k | P@5 | AUROC bg | AUROC loc | macro AUROC | **bg95 cost** |
|---|---|---|---|---|---|
| 2 | 0.999 | 0.995 | 0.861 | **0.876** (best) | 2.6% |
| **4** | 1.000 | 0.998 | 0.848 | 0.868 | **0.6%** ← selected |
| 8 | 0.999 | 0.993 | 0.819 | 0.853 | 3.6% |

**k = 4 confirmed, and the selection rule fires again.** AUROC would pick k=2; the operating
point picks k=4, at 4.4× lower cost. Same disagreement, same direction, independently of α —
which is the best evidence available that the rule is not a one-off artifact of the α axis.

k=8 is worst on both criteria: at 8× the box side the window is mostly unrelated scene, and
localisation discrimination degrades further (0.861 → 0.819) because the box occupies an ever
smaller fraction of the crop.

---

## `class_margin` fixed: real margin, but the relabel task is still saturated

**The feature was broken and is now fixed.** It used to be computed inside the top-5 window
with a −1.0 sentinel for "predicted class not among the neighbours". With purity ≈ 1.0 the
sentinel fired on every relabel, so the feature took **two** values, ±(1 + sim@1) — a sign
flip on agreement@5 wearing a margin's name. It is now computed over the whole positive pool:
best similarity to the predicted class minus best similarity to any other class.

| | before (top-5 + sentinel) | after (full memory) |
|---|---|---|
| distinct values on 1015 queries | 2 | **1011** |
| range | {−1.966, +1.961} | [−0.259, +0.183] |
| mean, accept | +1.961 | +0.124 |
| mean, relabel | −1.966 | −0.150 |
| mean, reject_localisation | +1.954 | +0.118 |
| mean, reject_background | −1.328 | −0.058 |

**But the relabel AUROC did not move off 1.000, and it should not be reported as a success.**
Accept's minimum margin is +0.0231 and relabel's maximum is +0.0171 — a real but razor-thin
0.006 gap with zero overlap across 659 queries. The separation is genuine; the *task* is
saturated. The cause is the synthetic construction, not the feature:

- Retrieval purity is 1.000, so any label disagreeing with a unanimous, high-similarity
  retrieval is caught by construction.
- A synthetic relabel is a correct object called the wrong thing. A real class error happens
  where the imagery is genuinely confusable, and no amount of relabelling clean pixels
  reproduces that.

Evidence the hardening did bite: `agreement_at_k`'s relabel AUROC fell from 1.000 to
**0.9961** once relabels moved onto jittered boxes with sampled target classes. The margin
still separates perfectly because it is a strictly better feature, not because the task got
easier.

**Do not quote relabel AUROC as evidence the gate catches class errors.** The real
class-error distribution needs detector proposals.

---

## §4 — how the relabels were hardened

Three changes, all in `src/retrieval/proposals.py`:

1. **Built on jittered boxes at IoU ≥ 0.5**, not clean GT. A real class error arrives with an
   imperfect box, and a pixel-perfect crop is the easiest possible case.
2. **Target class sampled from the measured confusion distribution**, not its argmax. Taking
   only the single most-confused class made every relabel of a given class identical, which
   both understates the variety of real errors and lets a gate learn a fixed pairing instead
   of the evidence. Classes absent from the confusion table fall back to a softmax over
   centroid cosine similarity.
3. **`accept` is now every clean box**; the jitter corpus supplies both reject_localisation
   (IoU < 0.5) and relabel (IoU ≥ 0.5). The three sources are disjoint by construction — each
   GT box has exactly one jitter — so nothing appears under two verdicts.

Resulting spread (was a single target per class):

| class | relabel targets sampled |
|---|---|
| airplane | storage_tank 0.70, tennis_court 0.22, vehicle 0.08 |
| ship | storage_tank 0.44, vehicle 0.32, harbor 0.16 |
| harbor | ship 0.48, tennis_court 0.37, airplane 0.15 |
| bridge | ground_track_field 0.44, baseball_diamond 0.44, harbor 0.12 |
| tennis_court | basketball_court 0.60, baseball_diamond 0.29, ground_track_field 0.08 |
| vehicle | storage_tank 0.59, ship 0.41 |
| ground_track_field | baseball_diamond 1.00 |

**Ceiling, stated plainly:** synthetic relabels cannot reach the difficulty of real ones. The
visual evidence is not genuinely ambiguous — the object is what it is, and only the label was
changed. Every relabel number here is an upper bound on real performance.

---

## Files

`alpha_sweep_summary.csv` · `alpha_sweep_auroc.csv` · `alpha_sweep_means.csv` ·
`tight_vs_local_per_class.csv` · `relabel_map.csv` · `proposals.csv` ·
`open_set_leave_one_class_out.csv` · `baseline_summary.csv` · `risk_coverage.csv` ·
`b2_weights.csv`

Reproduce: `python3 -m scripts.run_alpha_sweep`, then `run_open_set`, then `run_baselines`.
