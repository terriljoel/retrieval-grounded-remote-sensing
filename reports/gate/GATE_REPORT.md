# Phase 0 gate report — does retrieval work?

**Yes, decisively, and the project should proceed as planned.** Macro P@5 is 0.996 on clean
ground-truth crops against a 0.100 chance floor, and 0.991 when every query box is deliberately
mis-localised to IoU 0.4–0.7 with truth. The winning encoder is **RemoteCLIP ViT-L/14 on the local
stream**. Retrieval also survives the two tests that were added because the clean test was too easy:
it holds up under detector-grade localisation error, and it separates real objects from background
crops well enough for `novelty = 1 − sim@1` to remain a usable gate feature.

Scope: 555 positive images from train+val, 3,373 GT boxes. The test split was never touched.
Every query excludes all crops from its own source image; that exclusion removed a mean of 15.8
candidates per query (min 1, max 67).

> **Naming.** The two streams are called **local** and **regional**, not "object" and "context".
> §7 measures why: for elongated classes the local crop is roughly half surroundings already
> (harbor 0.47 box fill), so "object crop" claims a separation the geometry does not deliver. What
> is defensible is the scale each covers — regional spans ~16× the area of local at k=4. Numbers in
> this report are unchanged; only the labels are.

---

## 0. Read this before the tables: clean GT numbers cannot rank encoders

All four encoders score between 0.974 and 0.996 macro P@5 on clean GT crops. That 0.022 spread is
the measurement saturating, not four encoders being equal. Ranking anything on it would be reading
noise. The two added corpora exist to break that ceiling, and they do:

| corpus | metric | spread across the four encoders |
|---|---|---|
| clean | macro P@5 | 0.022 |
| jitter | macro P@5 | 0.035 |
| background | novelty AUROC | **0.190** |

The novelty test is roughly nine times more discriminating than clean purity. **Every encoder claim
in this report is made on the jitter and background corpora. The clean column is a reference row.**

---

## 1. Does retrieval work?

| encoder | stream | clean macro P@5 | jitter macro P@5 | macro chance | lift (jitter) |
|---|---|---|---|---|---|
| **remoteclip_l14** | local | 0.996 | **0.991** | 0.100 | **+0.891** |
| remoteclip_b32 | local | 0.992 | 0.981 | 0.100 | +0.881 |
| dinov2_b14 | local | 0.975 | 0.964 | 0.100 | +0.864 |
| openclip_b32 | local | 0.974 | 0.956 | 0.100 | +0.856 |
| remoteclip_b32 | regional | 0.958 | 0.943 | 0.100 | +0.843 |
| remoteclip_l14 | regional | 0.946 | 0.936 | 0.100 | +0.836 |
| openclip_b32 | regional | 0.879 | 0.865 | 0.100 | +0.765 |
| dinov2_b14 | regional | 0.843 | 0.830 | 0.100 | +0.730 |

The chance floor is drawn, not computed — five random eligible neighbours per query, seeded, carrying
the same same-image exclusion as the real measurement.

**Localisation error barely touches class purity.** Jitter costs the winner 0.005 macro P@5, and
purity is flat across IoU bands:

| IoU with truth | n | P@5 (l14/local) | mean sim@1 |
|---|---|---|---|
| 0.4–0.5 | 1,061 | 0.993 | 0.951 |
| 0.5–0.6 | 1,163 | 0.992 | 0.956 |
| 0.6–0.7 | 1,149 | 0.996 | 0.960 |
| 1.0 (clean) | 3,373 | 0.996 | 0.966 |

This was expected to be where encoders separate. It is not. A box half-off an aircraft still contains
an aircraft, and these ten classes are visually distinct enough that class survives. **The finding is
that class retrieval is robust to localisation error — and, as §6 explains, that is also a warning
about what the gate cannot detect.**

---

## 2. Which encoder wins, and does domain pretraining pay for itself?

**RemoteCLIP ViT-L/14, local stream, wins on every hard metric.**

The control comparison is exact — `remoteclip_b32` and `openclip_b32` are the same ViT-B/32
architecture on the same crops, differing only in pretraining:

| metric (local stream) | RemoteCLIP B/32 | OpenAI CLIP B/32 | delta |
|---|---|---|---|
| clean macro P@5 | 0.992 | 0.974 | +0.018 |
| jitter macro P@5 | 0.981 | 0.956 | +0.025 |
| novelty AUROC | 0.936 | 0.868 | **+0.068** |
| real objects lost buying 95% background rejection | 33.1% | 42.2% | **−9.1 pp** |

**Yes, remote-sensing pretraining pays for itself**, and the margin widens as the task gets harder —
on clean crops it looks like a rounding difference, on novelty it is decisive. OpenAI CLIP's problem
is visible in the raw scores: its background crops score sim@1 0.906 against 0.943 for real objects.
Everything is similar to everything in that space, so nothing thresholds.

**DINOv2 does not beat both.** It edges OpenAI CLIP on purity (0.964 vs 0.956 jitter) and is the
worst of the four on novelty by a wide margin (AUROC 0.791, costing 63.4% of real objects to reject
95% of background). The expectation that a self-supervised instance-retrieval model would win here
is not supported.

Scaling within RemoteCLIP is worth the compute: L/14 over B/32 buys +0.010 jitter P@5 and, far more
importantly, cuts the false-reject cost from 33.1% to **8.8%**.

---

## 3. Which classes work and which fail?

`remoteclip_l14`, local stream, jitter corpus, sorted by lift:

| class | n | P@1 | P@5 | chance | lift | clean P@5 |
|---|---|---|---|---|---|---|
| ground_track_field | 138 | 1.000 | 1.000 | 0.032 | +0.968 | 1.000 |
| bridge | 108 | 1.000 | 0.996 | 0.033 | +0.963 | 0.996 |
| harbor | 198 | 1.000 | 1.000 | 0.063 | +0.937 | 1.000 |
| basketball_court | 130 | 0.969 | 0.965 | 0.051 | +0.914 | 0.983 |
| ship | 254 | 0.972 | 0.976 | 0.077 | +0.898 | 0.993 |
| baseball_diamond | 334 | 1.000 | 0.993 | 0.097 | +0.896 | 0.998 |
| tennis_court | 454 | 0.993 | 0.992 | 0.131 | +0.860 | 0.996 |
| vehicle | 516 | 0.998 | 0.995 | 0.147 | +0.848 | 0.996 |
| storage_tank | 585 | 1.000 | 0.999 | 0.175 | +0.824 | 1.000 |
| airplane | 656 | 0.999 | 0.999 | 0.195 | +0.804 | 1.000 |

**No class fails.** The four classes the detector handles worst — tennis_court (29% correct), harbor
(18%), basketball_court (37%), ship (38%) — are all classes where retrieval is strong:

| class | detector correct | retrieval P@5 (l14/local/jitter) |
|---|---|---|
| harbor | 18% | 1.000 |
| tennis_court | 29% | 0.992 |
| basketball_court | 37% | 0.965 |
| ship | 38% | 0.976 |

This is the best possible result for the project's premise: **the verification layer is strongest
exactly where the detector is weakest.** basketball_court is the weakest of the four and is also the
class where encoder choice matters most (0.965 for l14 vs 0.806 for OpenAI CLIP), which is a further
argument for the larger RemoteCLIP.

Confusion is concentrated in two semantically sensible pairs (l14/local/jitter, top-5 neighbours):

| query class | wrong neighbour | count | share of all top-5 |
|---|---|---|---|
| ship | vehicle | 28 | 0.166% |
| basketball_court | tennis_court | 23 | 0.136% |
| tennis_court | basketball_court | 9 | 0.053% |
| vehicle | storage_tank | 9 | 0.053% |

Small elongated objects on uniform ground (ship/vehicle) and rectangular painted courts
(basketball/tennis) are what the VLM stage will need to disambiguate. Nothing else is close.

---

## 4. Is the regional stream redundant?

**No. Keep both streams.** Mean cosine between a crop's own local and regional embedding:

| encoder | paired cosine | neighbour-ranking correlation | redundant (>0.90)? |
|---|---|---|---|
| remoteclip_b32 | 0.704 | 0.665 | no |
| remoteclip_l14 | 0.761 | 0.610 | no |
| openclip_b32 | 0.775 | 0.546 | no |
| dinov2_b14 | 0.527 | 0.608 | no |

All well below the 0.90 redundancy line, so no masked-local variant is needed. But the streams are
not interchangeable, and their division of labour is sharper than expected:

- **Local wins on class purity everywhere** (0.991 vs 0.936 for l14).
- **Regional is near-perfect at rejecting whole-tile background** — l14 regional separates crops from
  the negative image set at AUROC **0.997**, costing only **1.7%** of real objects.
- **Regional is the worse of the two at rejecting empty patches inside positive tiles**
  (AUROC 0.898 vs 0.978 for local).

That is a genuine complementarity and it is the thing the Phase 1 α-sweep should be tuned on. One
caveat below in §7: the local crop is not a pure object view, which is why it is no longer called one.

---

## 5. Is there a hidden sensor split?

**Yes, and the box-area histogram is the wrong instrument to find it.**

31 of 555 images (the contiguous ID block 422–453) are Vaihingen colour-infrared; 524 are RGB.
This was found by colour statistics, not area: p99(R−G) separates the two with an empty gap — RGB
tiles span [−20, 84], CIR tiles span [98, 160], nothing in between. Confirmed visually at both edges
of the gap.

The §8.2 premise — that a hidden sensor split shows up as bimodal box areas — does not hold here:

| class | n | Otsu split | separation | interpretation |
|---|---|---|---|---|
| vehicle | 516 | 2,072 px² | 0.65 | CIR and RGB areas **overlap heavily** (medians 1,674 vs 2,668 px²) |
| ship | 254 | 3,015 px² | 0.64 | single mode, all RGB |
| storage_tank | 585 | 2,098 px² | 0.76 | genuinely two modes — but **both are RGB**, small vs large tanks |

For calibration: the known-bimodal p99(R−G) statistic only scores 0.729 on this same measure, and a
unimodal Gaussian control scores 0.637. At this sample size Otsu separation cannot distinguish one
mode from two, so no area-based conclusion is drawn. `box_area_hist.png` shows the overlap directly.

**The confound this diagnostic was meant to catch is real and was located precisely: all 365 CIR
crops are `vehicle`, and 70.7% of all vehicle crops are CIR.** Inside CIR, sensor matching *is* class
matching. Two independent checks say the sensor is not doing the work:

- RGB vehicle queries only (n=151, l14/local): P@5 **0.993**, lift +0.852 — indistinguishable from
  the CIR queries' 0.997.
- A separate run on an RGB-only sub-corpus, where no CIR crop exists to match against at all
  (checkpoint 3, remoteclip_b32/local, n=3,008): vehicle P@5 **0.985** against chance 0.052,
  lift **+0.934**.

Vehicle retrieval is genuinely good. Reporting convention: **vehicle numbers are RGB-only unless labelled
otherwise**, and `purity_by_sensor.csv` carries the stratified version.

---

## 6. Novelty: does `1 − sim@1` survive as a gate feature?

**Yes, but only with RemoteCLIP L/14, and only for the failure mode it actually detects.**

999 background crops — 500 from the negative image set, 499 from regions of positive images touching
no GT box — sized from the real GT area distribution so that size alone cannot separate them.
Queried against the same GT memory with the same same-image exclusion.

| encoder | stream | object sim@1 | background sim@1 | AUROC | cost of 95% bg rejection |
|---|---|---|---|---|---|
| **remoteclip_l14** | **local** | 0.966 | 0.888 | **0.981** | **8.8%** |
| remoteclip_l14 | regional | 0.946 | 0.853 | 0.948 | 36.2% |
| remoteclip_b32 | regional | 0.943 | 0.800 | 0.940 | 41.6% |
| remoteclip_b32 | local | 0.952 | 0.859 | 0.936 | 33.1% |
| openclip_b32 | local | 0.943 | 0.906 | 0.868 | 42.2% |
| openclip_b32 | regional | 0.929 | 0.888 | 0.854 | 57.4% |
| dinov2_b14 | local | 0.776 | 0.641 | 0.791 | 63.4% |
| dinov2_b14 | regional | 0.840 | 0.792 | 0.644 | 82.8% |

**The feature lives.** At sim@1 < 0.944, RemoteCLIP L/14's local stream rejects 95% of background
while losing 8.8% of real objects. That is a usable operating point for a gate whose whole purpose is
to route confidently-wrong proposals away from the human. With any other encoder the same rejection
rate costs a third to two-thirds of the real objects, which would be useless — **this feature is not
encoder-agnostic, and choosing DINOv2 or plain CLIP would have killed it.**

Background source matters, and the two streams fail in opposite directions:

| encoder / stream | negative-image AUROC (cost) | positive-free AUROC (cost) |
|---|---|---|
| remoteclip_l14 / local | 0.984 (6.4%) | **0.978 (10.3%)** |
| remoteclip_l14 / regional | **0.997 (1.7%)** | 0.898 (53.0%) |
| remoteclip_b32 / regional | 0.993 (1.7%) | 0.887 (57.1%) |

Object-free patches *inside* a positive tile are the hard case for the regional stream, and obviously
so — an empty crop from an airport tile still has apron and runway in its 4× regional window, which is
exactly what an aircraft's context looks like. The local stream does not care, because it is not
looking at the surroundings. L/14's local stream is the only configuration that handles both.

### The limitation that matters for the gate design

`sim@1` detects **"there is no object here"**. It does **not** detect **"the object is there but the
box is wrong"**:

| query type (l14/local) | mean sim@1 | drop from clean |
|---|---|---|
| clean GT box | 0.966 | — |
| jittered box, IoU 0.4–0.5 | 0.951 | −0.015 |
| background crop | 0.888 | −0.078 |

A badly localised box scores five times closer to a perfect box than a background crop does. So
novelty is a strong false-positive detector and a weak localisation-quality signal. **The gate needs
a separate feature for localisation quality — detector objectness or an IoU-prediction head — and it
cannot be recovered from retrieval similarity.** Worth knowing now rather than in week 6.

---

## 7. Local crops are half surroundings, which confounds the α-sweep

Mean fraction of the local window actually occupied by the box. The ceiling is 0.826 — squaring on
the longer side and padding 10% means even a perfect square box gives up 17%:

| class | mean box fill | flag |
|---|---|---|
| storage_tank | 0.773 | |
| airplane | 0.720 | |
| baseball_diamond | 0.706 | |
| tennis_court | 0.596 | **LOW** |
| basketball_court | 0.592 | **LOW** |
| ground_track_field | 0.591 | **LOW** |
| vehicle | 0.566 | **LOW** |
| bridge | 0.553 | **LOW** |
| ship | 0.505 | **LOW** |
| harbor | 0.468 | **LOW** |

**Seven of ten classes fall below 0.6, and the prediction that ship and bridge would be roughly half
background is confirmed** (0.505 and 0.553). For harbor, less than half the local crop is the object itself.

The consequence is real: the local and regional streams are already entangled by construction for
elongated classes, so their measured correlation (§4) understates how much they share, and a Phase 1
α-sweep that reads as "regional helps harbor" may be measuring nothing more than the local crop
already containing harbor surroundings. **Resolved in Phase 1**: the geometry is kept (masking and
aspect-preserving resizing both feed the encoder out-of-distribution input, and the local stream's
0.981 novelty AUROC is not worth risking), the streams are renamed for the scale they cover rather
than for what they are assumed to contain, and a `local_tight` control arm — the box resized straight
to 224, accepting distortion — quantifies the entanglement inside the α-sweep.

---

## 8. Recommendation

**Proceed as planned. No scope narrowing.** All ten classes clear the gate with lift above +0.80
under localisation noise, including all four of the detector's worst classes.

Committed choices from this test:

1. **Encoder: `remoteclip_l14`.** It wins purity, and on the novelty feature it is not merely better
   but the difference between a usable gate feature (8.8% false-reject) and an unusable one.
2. **Both streams retained.** Correlation 0.76 is well under the redundancy line, and the two have
   complementary background-rejection profiles.
3. **`novelty = 1 − sim@1` stays as a gate input**, scoped to false-positive detection. It is not a
   localisation-quality signal, and a separate feature is needed for that.
4. **Vehicle is reported RGB-only by default**, sensor-stratified always available.

One open item is listed in `docs/STATE.md`: the fact that
the whole test uses GT boxes — the detector's real proposal distribution (score-thresholded, with its
own class errors) is still unmeasured and depends on the export path.

---

## Artifacts

| file | contents |
|---|---|
| `crop_index.csv` | 3,373 clean crops: geometry, sensor, upsample factor, box fill |
| `jitter_index.csv` | 3,373 jittered crops with achieved IoU and band |
| `background_index.csv` | 999 background crops with source |
| `purity_per_class.csv` | per class × encoder × stream × corpus, with same-sensor columns |
| `purity_by_iou.csv` | P@5 and sim@1 by IoU band |
| `novelty.csv` | sim@1 distributions, AUROC, thresholds, costs, by background source |
| `purity_by_size.csv` | COCO buckets, area tertiles, upsample buckets |
| `purity_by_sensor.csv` | per-class P@5 split by query sensor |
| `obj_context_correlation.csv` | §8.1 redundancy test |
| `confusion_pairs.csv` | top-10 wrong-neighbour pairs per encoder/stream/corpus |
| `box_fill_by_class.csv` | §7 |
| `box_area_hist.png`, `umap_*.png`, `contact_sheet.jpg` | figures |

Reproduce with `python3 -m scripts.run_gate_test` (seed 42; the repo is not pip-installed locally, so
the `-m` form is required).
