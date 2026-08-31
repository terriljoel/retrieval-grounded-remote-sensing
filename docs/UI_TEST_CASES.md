# UI test cases — `app/gate_explorer.py`

**Re-walked 2026-08-31** after two changes: (1) `NIM_MODEL` moved to
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (the old model was retired,
HTTP 410) with a corrected VLM scoring path (see "VLM scoring: fixed"
below), and (2) `NOVELTY_ACCEPT_THRESHOLD` recalibrated on real data,
0.9436 → **0.9601** (see `docs/STATE.md`, "Threshold recalibration" — the
recalibrated value is *stricter*, not more permissive; it was expected to
fix case B below and instead makes it more pronounced). Driven
programmatically through `load_context`/`similar_cases`/`vlm_verdict`,
same functions and cache keys the UI itself uses.

## Prerequisites

```
export NIM_API_KEY='nvapi-...'          # or source .env, see .env.example
streamlit run app/gate_explorer.py
```

If `detections.csv` is missing, the app shows the exact fix on the page
(`load_context()` raises `FileNotFoundError`) rather than a stack trace.

## VLM scoring: fixed, and now working live in the app

The previous walkthrough found the old model retired and stopped there.
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` works, but is a *reasoning*
model — it opens every response with a `<think>...</think>` block before
answering, so the old scoring path (`yes_probability(entries[0])`, which
assumed the first generated token was the answer) breaks silently: it
would have read logprobs from the model's opening reasoning word, not its
answer. **Near-miss, not caught by any exception**: a naive fix (scan
forward for the first yes/no-shaped token) also fails silently, in a
different way — the reasoning trace routinely contains the literal phrase
"a yes/no question" while restating the question, so a forward scan
matches "yes" *inside that phrase* on a majority of crops tested (3 of 5 in
initial sampling), producing a plausible-looking but meaningless score. The
actual fix scans from the **end** of the sequence: the model reliably
restates its answer a second time, cleanly, immediately after `</think>`,
and that is what `find_answer_position` (`scripts/run_b3_vlm.py`) now
reads. Validated on 20 real escalated-tail crops: 20/20 readable, correct
position every time. This belongs alongside the QuickGELU and class_margin
near-misses as the same failure class: **plausible output, no error
raised** — see `docs/STATE.md`'s near-misses note.

Operational caveat: this model does not honour `max_tokens` on this
endpoint (confirmed: requests of 1, 10, 50 all returned 490–620 completion
tokens with `finish_reason="stop"`) and costs roughly 500+ completion
tokens per call regardless, versus the old model's ~1-token response. The
app's `vlm_verdict` step is correspondingly slower and more expensive per
click than before — still fine for a live demo (one click, one call), just
worth knowing.

## What each step actually shows now

1. **Image dropdown** — unchanged, 107/120 val images have a detection.
2. **Tile with boxes** — unchanged.
3. **Detection dropdown** — unchanged, confirmed correct.
4. **"Find similar cases"** — unchanged, working.
5. **"Ask the VLM"** — **now works.** All four cases below returned
   `p(yes)=1.0000` — including case A, a real false positive. Worth
   knowing going into a demo: in this small sample the model did not
   discriminate at all (100% "yes"); the 20-crop validation set separately
   showed 16 "yes" / 4 "no", so the model is not a rubber stamp in
   general, just uniformly confident on these four particular crops (all
   of which do have strong, class-consistent retrieval evidence backing
   them, including the false positive and the annotation-gap case — both
   look genuinely plausible, which is rather the point of cases A and C).
6. **Gate verdict vs ground truth** — now escalates far more often. With
   the stricter threshold, **3 of 4 cases below now escalate**, including
   two (B, D) that are real, correctly-localised, ground-truth accepts.

## The four cases, re-run with the new threshold

**A — background false positive.** `detector:nwpu_background_053_det000`
(`ship`, confidence 0.190). top-1 sim 0.933. Gate verdict: **escalate**
(0.933 < 0.9601 — unchanged from before, both thresholds agree here).
VLM: p(yes)=1.0000 despite this being a false positive — the retrieval
evidence (5/5 neighbours agree "ship") and the VLM agree with the
detector's wrong claim; nothing in this pipeline catches this specific
false positive.

**B — confident, correct detection.** `detector:nwpu_positive_309_det000`
(`storage_tank`, confidence 0.985, IoU 0.85, ground truth `accept`). top-1
sim 0.931. Gate verdict: **escalate** (0.931 < 0.9601 — was already wrong
at the old threshold, 0.9436; the new, "corrected" threshold does not fix
this, it just isn't the reason this case escalates). VLM: p(yes)=1.0000
(correct). **This is now the expected, typical outcome, not an edge
case** — the recalibrated threshold is stricter across the board.

**C — NWPU annotation-gap case.** `detector:nwpu_positive_170_det000`
(`ground_track_field`, confidence 0.940, top-1 sim 0.949, 5/5 agree,
ground truth `reject_background`, `iou=0.0`). Gate verdict: **escalate**
— **this flipped from `accept` under the old threshold** (0.949 ≥ 0.9436)
to `escalate` under the new one (0.949 < 0.9601). Arguably the more
sensible outcome for this specific case: an annotation gap is exactly the
kind of ambiguous case that should reach a human, not auto-accept. VLM:
p(yes)=1.0000 — the model was shown a real ground_track_field and,
reasonably, says yes; it has no way to know NWPU's GT never boxed it.

**D — confusable-class case.** `detector:nwpu_positive_081_det003`
(`tennis_court`, confidence 0.906, IoU 0.78, ground truth `accept`,
top-1 sim 0.950, 5/5 agree). Gate verdict: **escalate — this also
flipped**, from `accept` (0.950 ≥ 0.9436) to `escalate` (0.950 < 0.9601).
Unlike case C, this is a **new disagreement with ground truth** that did
not exist before recalibration: a correct, well-localised, class-
consistent detection now gets sent to a human. VLM: p(yes)=1.0000
(correct).

**Net effect of recalibration on these four cases**: A unchanged, C
improved (arguably), B unchanged-but-now-typical, D newly wrong. Not a
clean win — matches `docs/STATE.md`'s finding that recalibration trades
lower background leakage for a large real-accept coverage loss (63.4% →
27.4% at a matched 1% leak budget), and this small sample shows exactly
that trade in miniature: 2 of 3 non-background cases now escalate.

## If a demo goes sideways

- **Page shows a `FileNotFoundError` naming `detections.csv`**: the export
  isn't at the configured path. See Prerequisites.
- **"Ask the VLM" raises a `RuntimeError` naming `NIM_API_KEY`**: the key
  isn't in the environment — `export NIM_API_KEY=...` or
  `set -a; . ./.env; set +a` first.
- **"Ask the VLM" is slow (several seconds)**: expected — this model
  generates a full reasoning trace (typically 300–600 tokens) before
  every answer, and doesn't honour `max_tokens`. Not a hang.
- **A similar-case thumbnail doesn't render**: `crop_path_for_case` derives
  the file from `proposal_id.split(":", 1)`; check `data/memory/cases.db`
  if this happens.
- **The same detection gives a different similarity or VLM score on a
  second click**: should not happen (`@st.cache_data` keyed on
  `proposal_id`) — if it does, suspect the `force=True` re-embed in
  `embed_proposal`, or a genuinely non-deterministic model response (this
  reasoning model is called at `temperature=0.0`, so it should be stable,
  but reasoning-trace length can still vary run to run on some endpoints).

## Overall gate behaviour to expect (context, not a demo script)

Everything under `reports/FINDINGS.md`'s `[SYN]` tag describes the
synthetic-proposal gate, not this app's real detections. The B0-vs-B2
findings and the threshold recalibration findings (`docs/STATE.md`) are
the real-data numbers, and case D above now demonstrates the same thing
they do: neither the synthetic-fit B2 gate nor a real-data-recalibrated
similarity threshold reliably tracks real correctness — expect the demo to
surface disagreements on well-behaved cases, not just on hard ones.
