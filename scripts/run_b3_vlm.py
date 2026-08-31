"""B3: a hosted VLM adjudicating the escalated band, on the class-error decision only.

B2 auto-accepts the head of its ranking and escalates the tail. B3 leaves that
head exactly as it is and re-ranks **only the escalated tail**, by asking a
vision model "does this crop show a <predicted_class>?".

    B3 ranking = [head, ordered by B2] ++ [tail, ordered by the VLM]

The number to beat is B2's coverage at a 1% class-error budget. If the VLM can
find the correct-class proposals inside the escalated band, that coverage goes
up; if it cannot, the curve is unchanged or worse and B3 has not earned its
inference cost.

Scope, deliberately: the VLM is asked about class only. It sees the same local
crop retrieval sees, and BASELINES.md establishes that this crop does not carry
box-tightness information; asking it to judge localisation would buy confident
noise. It is also never run on the auto-accepted head, which is already inside
budget and which no answer could improve.

**`--scoring` has no default, on purpose.** A risk-coverage curve needs a
continuous score, and which mechanism produces one depends on what the endpoint
returns. Run `scripts/probe_nim.py` first; it answers that empirically. Nothing
here silently falls back to parsing a generated "yes".
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.retrieval.fuse import TOP_K, evidence, fused_similarity
from src.retrieval.gate import auroc
from src.retrieval.nim import (
    call, encode_image, model_name, request_body, samples as env_samples,
    temperature as env_temperature, usage_report,
)
from src.retrieval.phase1 import ENCODER, MAIN_SCALE, SEED, prepare, selected_alpha
from src.utils.paths import find_project_root
from scripts.run_baselines import (
    FALSE_ACCEPT_BUDGETS, HARMS, HOLDOUT_FRACTION, aurc, build_features,
    coverage_at_budget, risk_coverage, stability_available,
)

# Model, temperature and sample count are read from NIM_MODEL /
# NIM_TEMPERATURE / NIM_SAMPLES. Nothing here hardcodes a model id: switching
# arms is an environment change, and the two-model comparison is the point.
TARGET_HARM = "class_error"
TARGET_BUDGET = 0.01
SCORING = ("logprob", "self_consistency", "elicited")

# Outer, patient retry for a long unattended run (see score_one_patient) --
# deliberately much slower than nim.call()'s own inner backoff (2s base,
# 60s cap, 6 attempts): a 429/503 storm needs minutes to clear, not seconds.
PATIENT_MAX_ATTEMPTS = 40
PATIENT_BASE_DELAY = 30.0    # seconds
PATIENT_BACKOFF_CAP = 300.0  # 5 minutes between retries, capped
# A response the endpoint delivered successfully but that this scoring
# method cannot read (no logprobs at all, or no yes/no token anywhere in the
# generated sequence). Distinct from RuntimeError, which nim.call() raises
# for transport failures (429/503/timeouts) -- those are worth a long patient
# retry; this is not. At temperature=0.0 an unscorable response will almost
# certainly reproduce itself on retry, so hammering it with the same 40-
# attempt/5-minute-cap schedule as a network error wastes hours per crop for
# nothing (measured directly: two crops each burned ~3 hours and ~35
# uncached API calls this way before this distinction existed).
UNSCORABLE_RETRIES = 1        # one quick retry only, for rare genuine variance
UNSCORABLE_RETRY_DELAY = 5.0  # seconds


class UnscorableResponse(RuntimeError):
    pass

YES_NO_PROMPT = (
    "This is an aerial or satellite image crop. "
    "Does it show a {label}? Answer only yes or no."
)
# The ungrounded prompt above shows the VLM no retrieval evidence at all, so a
# counterfactual-evidence ablation against it would flip something the model
# never sees and measure nothing by construction. This variant puts the
# retrieved cases in the prompt so the question can actually be asked.
GROUNDED_PROMPT = (
    "This is an aerial or satellite image crop. "
    "A database of previously human-verified examples was searched, and the most "
    "similar verified cases are:\n{evidence}\n"
    "Does this crop show a {label}? Answer only yes or no."
)
EVIDENCE_MODES = ("none", "true", "counterfactual")
ELICITED_PROMPT = (
    "This is an aerial or satellite image crop. "
    "How confident are you that it shows a {label}? "
    "Answer with only an integer from 0 to 100."
)


def readable(class_name: str) -> str:
    """`storage_tank` -> `storage tank`; the model was not trained on our slugs."""
    return class_name.replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--model", default=None,
                        help="Overrides NIM_MODEL for this run.")
    parser.add_argument("--evidence", choices=EVIDENCE_MODES, default="none",
                        help="none: no retrieval in the prompt (the B3 of record). "
                             "true: the real top-k retrieved cases. "
                             "counterfactual: the same cases relabelled wrong.")
    parser.add_argument("--scoring", choices=SCORING, required=True,
                        help="How to get a continuous score. No default: run "
                             "scripts/probe_nim.py first to see what the endpoint supports.")
    parser.add_argument("--samples", type=int, default=None,
                        help="Draws for --scoring self_consistency.")
    parser.add_argument("--estimate", action="store_true",
                        help="Report call and token cost for the escalated band, then stop.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N escalated crops (smoke run).")
    parser.add_argument("--setup-limit", type=int, default=None,
                        help="Limit the corpora themselves; for pipeline smoke runs.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel requests for --scoring logprob (reasoning models "
                             "run ~8s/call; sequential doesn't finish in reasonable time). "
                             "Each call still goes through nim.call()'s own retry/backoff, "
                             "so a higher value just means more calls in flight, not less "
                             "backoff. Ignored for other --scoring modes.")
    return parser.parse_args()


YES_TOKENS = ("yes", "y")
NO_TOKENS = ("no", "n")
TOP_LOGPROBS = 20


def yes_probability(entry: dict) -> float:
    """p(yes) from one token position's top_logprobs.

    Endpoints differ in how much of the distribution they return, and the
    difference is not cosmetic. nemotron-nano-12b returns the full top-20, so
    both branches are present and the score can be renormalised over {yes, no}
    -- which is the right thing, because it discards mass on irrelevant tokens
    like "The".

    llama-3.1-nemotron-nano-vl-8b returns only the argmax. Renormalising there
    would divide by the one branch present and hand back 1.0 for every single
    call, which looks like a working feature and is in fact a constant. So when
    only one branch is visible the **absolute** probability of that token is
    used instead: p(yes) directly, or 1 - p(no). Both are monotone in
    yes-ness, so the ranking within an arm stays coherent, which is all a
    risk-coverage curve needs.
    """
    yes = no = 0.0
    for candidate in entry.get("top_logprobs") or []:
        token = candidate["token"].strip().lower()
        if token in YES_TOKENS:
            yes += float(np.exp(candidate["logprob"]))
        elif token in NO_TOKENS:
            no += float(np.exp(candidate["logprob"]))
    if yes > 0.0 and no > 0.0:
        return yes / (yes + no)
    if yes > 0.0:
        return min(yes, 1.0)
    if no > 0.0:
        return max(1.0 - no, 0.0)
    raise RuntimeError(f"Neither yes nor no among the returned logprobs: {entry}")


def counterfactual_classes(class_names, predicted, verified, count, rng):
    """Labels that are neither the claimed class nor the true one.

    Excluding the *true* class as well as the claimed one is what makes this a
    control rather than a second relabel task. If the true class could appear,
    a counterfactual on a wrong-class proposal would sometimes hand the model
    the correct answer, and the arm would look informative for the opposite of
    the intended reason.
    """
    forbidden = {predicted, verified}
    pool = [name for name in class_names if name not in forbidden]
    if not pool:
        raise ValueError(f"No counterfactual class available outside {forbidden}")
    return rng.choice(pool, size=count)


def evidence_block(classes, similarities) -> str:
    return "\n".join(
        f"  - a verified {readable(name)} (similarity {score:.2f})"
        for name, score in zip(classes, similarities)
    )


def build_prompt(label: str, evidence_text: str | None) -> str:
    if evidence_text is None:
        return YES_NO_PROMPT.format(label=readable(label))
    return GROUNDED_PROMPT.format(label=readable(label), evidence=evidence_text)


# A generous ceiling, not a real budget: nemotron-3-nano-omni-...-reasoning
# ignores max_tokens entirely on this endpoint (confirmed empirically --
# max_tokens=1, 10 and 50 all returned 490-620 completion tokens with
# finish_reason="stop"), so there is no dial here that actually controls cost
# or truncation. Set high enough to never truncate a real response rather
# than to bound one.
REASONING_MAX_TOKENS = 700


def find_answer_position(entries: list[dict]) -> int | None:
    """Index of the *last* generated token that is itself yes/no, case-insensitive.

    A direct-answer model's first token is the answer, so index 0 always
    worked (the old scoring path). A reasoning model's response is
    `<think>...restates the question, often using the literal phrase "a
    yes/no question"...</think>\\nYes` -- so the FIRST yes/no-shaped token is
    frequently a false positive from that framing, not the answer (measured
    directly: 3 of 5 sample crops matched inside "yes/no question" at
    position 5). The genuine answer is reliably the token immediately after
    `</think>`, which is also the last such token in the sequence -- so scan
    from the end.
    """
    for index in range(len(entries) - 1, -1, -1):
        token = (entries[index].get("token") or "").strip().lower()
        if token in YES_TOKENS or token in NO_TOKENS:
            return index
    return None


def score_logprob(image_url: str, label: str, model: str, cache: Path,
                  evidence_text: str | None = None,
                  max_tokens: int = REASONING_MAX_TOKENS) -> float:
    """p(yes) from the logprobs at wherever the model actually answers.

    `max_tokens=1` (the old default) assumed the first generated token was
    the answer. That held for direct-answer models; a reasoning model puts a
    `<think>` block first, so `max_tokens` is raised and the answer token is
    *located* (`find_answer_position`) rather than assumed to be at index 0.
    """
    payload = call(request_body(
        model, build_prompt(label, evidence_text), image_url,
        temperature=0.0, max_tokens=max_tokens, logprobs=True, top_logprobs=TOP_LOGPROBS,
    ), cache_root=cache)
    entries = (payload["choices"][0].get("logprobs") or {}).get("content")
    if not entries:
        raise UnscorableResponse(
            "The endpoint returned no logprobs. Re-run scripts/probe_nim.py and "
            "pick a different --scoring mode; this one cannot work here."
        )
    position = find_answer_position(entries)
    if position is None:
        raise UnscorableResponse(
            f"No yes/no token found among {len(entries)} generated tokens for "
            f"this model/prompt -- it may not be answering yes/no at all, or "
            f"the response was truncated before it got there."
        )
    return yes_probability(entries[position])


def score_self_consistency(
    image_url: str, label: str, model: str, cache: Path, samples: int
) -> float:
    """Fraction of independent samples answering yes. Granularity is 1/samples.

    Temperature 0.7 rather than 1.0: hot enough that the model does not return
    the identical token every draw (which would make the score binary and the
    whole exercise pointless), cool enough that the variation reflects genuine
    uncertainty rather than sampling noise on a confident answer.
    """
    yes = 0
    for index in range(samples):
        payload = call(request_body(
            model, YES_NO_PROMPT.format(label=readable(label)), image_url,
            temperature=env_temperature(), max_tokens=4, seed=index,
        ), cache_root=cache)
        text = (payload["choices"][0].get("message") or {}).get("content") or ""
        yes += text.strip().lower().startswith("yes")
    return yes / samples


def score_elicited(image_url: str, label: str, model: str, cache: Path) -> float:
    """A number the model states about itself. Cheap, and poorly calibrated."""
    payload = call(request_body(
        model, ELICITED_PROMPT.format(label=readable(label)), image_url,
        temperature=0.0, max_tokens=8,
    ), cache_root=cache)
    text = (payload["choices"][0].get("message") or {}).get("content") or ""
    match = re.search(r"\d{1,3}", text)
    if not match:
        raise RuntimeError(f"No number in elicited confidence: {text!r}")
    return min(int(match.group()), 100) / 100.0


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = model_name()
    if args.samples is None:
        args.samples = env_samples()
    rng = np.random.default_rng(SEED)
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"
    crop_root = root / "data" / "crops"
    nim_cache = root / "data" / "nim_cache"

    alpha_fusion = args.alpha
    if alpha_fusion is None:
        alpha_fusion = selected_alpha(report_root / "alpha_sweep_summary.csv")
    print(f"B3: model={args.model} scoring={args.scoring} evidence={args.evidence} "
          f"samples={args.samples} temperature={env_temperature()} "
          f"fusion alpha={alpha_fusion} k={args.scale:g}")

    setup = prepare(root, args.encoder, args.scale, limit=args.setup_limit)
    train, val = setup.train, setup.val
    use_stability = stability_available(setup)

    # --- B2, exactly as in run_baselines ----------------------------------
    train_images = sorted({r["image_uid"] for r in train})
    shuffled = np.array(train_images)[rng.permutation(len(train_images))]
    holdout = set(shuffled[: int(len(shuffled) * HOLDOUT_FRACTION)].tolist())
    fit_memory = [r for r in train if r["image_uid"] not in holdout]
    fit_queries = [r for r in train if r["image_uid"] in holdout]

    fit_x, _ = build_features(setup, fit_queries, fit_memory, alpha_fusion, stability=use_stability)
    fit_y = np.array([r["verdict"] == "accept" for r in fit_queries])
    val_x, _ = build_features(setup, val, train, alpha_fusion, stability=use_stability)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(fit_x, fit_y)
    b2_scores = model.predict_proba(val_x)[:, 1]

    verdicts = np.array([r["verdict"] for r in val])
    harmful = verdicts == HARMS[TARGET_HARM]
    harmless = ~harmful

    # --- the escalated band -----------------------------------------------
    coverage, risk = risk_coverage(b2_scores, harmless)
    b2_coverage = coverage_at_budget(coverage, risk, TARGET_BUDGET)
    order = np.lexsort((np.random.default_rng(SEED).permutation(len(b2_scores)), -b2_scores))
    head_size = int(round(b2_coverage * len(val)))
    head, tail = order[:head_size], order[head_size:]
    if args.limit:
        tail = tail[: args.limit]
    print(f"\nB2 coverage at {TARGET_BUDGET:.0%} {TARGET_HARM}: {b2_coverage:.3f} "
          f"({head_size} auto-accepted, {len(tail)} escalated"
          f"{f', scoring {len(tail)} of them' if args.limit else ''})")
    names, counts = np.unique(verdicts[tail], return_counts=True)
    print("  escalated band: " + ", ".join(f"{n} {c}" for n, c in zip(names, counts)))

    # --- retrieval evidence for the prompt, if asked for -------------------
    evidence_text: dict[int, str] = {}
    if args.evidence != "none":
        positive_mask = np.array([r["verdict"] == "accept" for r in train])
        memory_verified = np.array([r["verified_class"] or "__background__" for r in train])
        similarity = fused_similarity(
            setup.gather(val, "local"), setup.gather(val, setup.regional),
            setup.gather(train, "local"), setup.gather(train, setup.regional), alpha_fusion,
        )
        retrieved = evidence(similarity, positive_mask, TOP_K)
        counter_rng = np.random.default_rng(SEED)
        for index in tail:
            classes = memory_verified[retrieved["positive_idx"][index]]
            scores = retrieved["positive_sim"][index]
            if args.evidence == "counterfactual":
                # Replace every retrieved label with one that is neither the
                # claimed class nor the true class, keeping the similarities.
                # If the VLM is a rubber stamp for retrieval this must flip it.
                classes = counterfactual_classes(
                    setup.class_names, val[index]["predicted_class"],
                    val[index]["verified_class"], len(classes), counter_rng,
                )
            evidence_text[index] = evidence_block(classes, scores)

    # --- cost, before anything is spent -----------------------------------
    per_crop = args.samples if args.scoring == "self_consistency" else 1
    calls = len(tail) * per_crop
    if args.estimate:
        print(f"\n=== cost estimate ===")
        print(f"  escalated crops       {len(tail)}")
        print(f"  calls per crop        {per_crop} ({args.scoring})")
        print(f"  total API calls       {calls}")
        try:
            record = val[tail[0]]
            path = crop_root / record["corpus"] / "local" / f"{record['crop_id']}.jpg"
            with Image.open(path) as image:
                image_url = encode_image(image)
            probe = call(request_body(
                args.model, YES_NO_PROMPT.format(label=readable(record["predicted_class"])),
                image_url, temperature=env_temperature(), max_tokens=4, seed=0,
            ), cache_root=nim_cache)
            usage = probe.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            print(f"  measured per call     {prompt_tokens} prompt + "
                  f"{completion_tokens} completion = {prompt_tokens + completion_tokens}")
            print(f"  ESTIMATED TOTAL       {calls * (prompt_tokens + completion_tokens):,} tokens "
                  f"({calls * prompt_tokens:,} prompt + "
                  f"{calls * completion_tokens:,} completion)")
        except RuntimeError as error:
            print(f"\n  Token counts need one live call, which needs a key:\n  {error}")
            print(f"  Call count above is exact regardless.")
        print("\n  --estimate set: stopping before spending the rest.")
        return

    # --- VLM on the escalated band only -----------------------------------
    print(f"\nScoring {len(tail)} crops with {args.model} ({calls} calls, "
          f"concurrency={args.concurrency})...")
    vlm_scores = np.empty(len(tail), dtype=np.float64)

    def score_one(position: int) -> float:
        index = tail[position]
        record = val[index]
        path = crop_root / record["corpus"] / "local" / f"{record['crop_id']}.jpg"
        with Image.open(path) as image:
            image_url = encode_image(image)
        label = record["predicted_class"]
        if args.scoring == "logprob":
            return score_logprob(
                image_url, label, args.model, nim_cache, evidence_text.get(index))
        if args.scoring == "self_consistency":
            return score_self_consistency(
                image_url, label, args.model, nim_cache, args.samples)
        return score_elicited(image_url, label, args.model, nim_cache)

    def score_one_patient(position: int) -> float:
        """`score_one`, but a transient failure never kills the run.

        `nim.call()` already retries each request up to MAX_ATTEMPTS times
        with its own short backoff; this is a second, much more patient
        layer on top, specific to long unattended runs -- a longer base
        delay, many more attempts, and it never raises. A crop still failing
        after PATIENT_MAX_ATTEMPTS is recorded as NaN (dropped from the
        analysis below, with a count printed) rather than losing hours of
        otherwise-completed, cached work on every other crop in the arm.
        Every successful call is written to `nim_cache` as it lands (see
        `nim.call`), so re-running this same command after a crash or a
        manual stop re-derives the escalated band deterministically and
        picks up for free wherever it left off.
        """
        last: Exception | None = None
        unscorable_attempts = 0
        network_attempts = 0
        while True:
            try:
                return score_one(position)
            except UnscorableResponse as error:
                # Deterministic at temperature=0.0 -- the network-retry
                # backoff below would just repeat the same non-answer for
                # hours. One quick retry only, then give up on this crop.
                last = error
                unscorable_attempts += 1
                if unscorable_attempts > UNSCORABLE_RETRIES:
                    print(f"\n  crop {position} UNSCORABLE (not a network error, "
                          f"not retrying further): {last}", flush=True)
                    return float("nan")
                time.sleep(UNSCORABLE_RETRY_DELAY)
            except RuntimeError as error:
                last = error
                network_attempts += 1
                if network_attempts > PATIENT_MAX_ATTEMPTS:
                    print(f"\n  crop {position} PERMANENTLY FAILED after "
                          f"{PATIENT_MAX_ATTEMPTS} network-retry attempts: {last}",
                          flush=True)
                    return float("nan")
                delay = min(PATIENT_BASE_DELAY * (2 ** (network_attempts - 1)),
                            PATIENT_BACKOFF_CAP)
                print(f"\n  crop {position} failed (attempt {network_attempts}/"
                      f"{PATIENT_MAX_ATTEMPTS}): {last} -- retrying in {delay:.0f}s",
                      flush=True)
                time.sleep(delay)

    if args.scoring == "logprob" and args.concurrency > 1:
        # Each call still goes through nim.call()'s own retry/backoff
        # independently; concurrency only shortens wall-clock, it doesn't
        # bypass rate limiting. A reasoning model's per-call latency (~8s,
        # dominated by the reasoning trace, not network time) is why this
        # exists at all -- sequential scoring of a few hundred crops does
        # not finish in a reasonable time otherwise.
        from concurrent.futures import ThreadPoolExecutor

        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for position, score in zip(
                range(len(tail)), pool.map(score_one_patient, range(len(tail)))
            ):
                vlm_scores[position] = score
                done += 1
                if done % 25 == 0:
                    print(f"\r  {done}/{len(tail)}", end="", flush=True)
    else:
        for position in range(len(tail)):
            vlm_scores[position] = score_one_patient(position)
            if (position + 1) % 25 == 0:
                print(f"\r  {position + 1}/{len(tail)}", end="", flush=True)
    print(f"\r  {len(tail)}/{len(tail)}")
    print(f"  {usage_report()}")

    failed = np.isnan(vlm_scores)
    if failed.any():
        print(f"\n  {int(failed.sum())}/{len(tail)} crops permanently failed and are "
              f"EXCLUDED from the analysis below (see the per-crop warnings above).")
        tail = tail[~failed]
        vlm_scores = vlm_scores[~failed]

    tail_verdicts = verdicts[tail]
    tail_harmful = tail_verdicts == HARMS[TARGET_HARM]
    tail_accepts = tail_verdicts == "accept"

    # The escalated band may contain no true accepts at all -- B2's head can
    # capture every one of them. When that happens the meaningful question is
    # not "accept vs relabel" but "relabel vs everything else in the band":
    # the VLM's job there is to sink the wrong-class proposals below the
    # right-class-but-wrong-box ones.
    print(f"\n  band composition: {dict(zip(*[list(x) for x in np.unique(tail_verdicts, return_counts=True)]))}")
    print(f"  true accepts in the escalated band: {int(tail_accepts.sum())}")
    if tail_harmful.any() and tail_accepts.any():
        standalone = auroc(vlm_scores[tail_accepts], vlm_scores[tail_harmful])
        b2_tail = auroc(b2_scores[tail][tail_accepts], b2_scores[tail][tail_harmful])
        print(f"  AUROC accept vs relabel in band: VLM {standalone:.4f} vs B2 {b2_tail:.4f}")
    elif tail_harmful.any():
        others = ~tail_harmful
        standalone = auroc(vlm_scores[others], vlm_scores[tail_harmful])
        b2_tail = auroc(b2_scores[tail][others], b2_scores[tail][tail_harmful])
        print(f"  p(yes): non-relabel mean {vlm_scores[others].mean():.4f}, "
              f"relabel mean {vlm_scores[tail_harmful].mean():.4f}")
        print(f"  AUROC relabel vs rest of band: VLM {standalone:.4f} vs B2 {b2_tail:.4f}")
    else:
        standalone = float("nan")

    # --- B3 = head unchanged, tail re-ranked ------------------------------
    if args.limit:
        print("\n--limit set: skipping the risk-coverage comparison, which needs "
              "the whole escalated band.")
        return

    b3_scores = np.empty(len(val), dtype=np.float64)
    # The head keeps its order and stays strictly above every tail score.
    b3_scores[head] = 1.0 + np.linspace(1.0, 0.0, len(head)) if len(head) else 0.0
    b3_scores[tail] = vlm_scores

    rows = []
    for label, scores in (("B2_logreg", b2_scores), ("B3_vlm_escalated", b3_scores)):
        curve_coverage, curve_risk = risk_coverage(scores, harmless)
        row = {"baseline": label, "harm": TARGET_HARM, "model": args.model,
               "scoring": args.scoring, "aurc": round(aurc(curve_risk), 4)}
        for budget in FALSE_ACCEPT_BUDGETS:
            row[f"coverage_at_{budget:.0%}"] = round(
                coverage_at_budget(curve_coverage, curve_risk, budget), 4)
        rows.append(row)

    print(f"\n{'baseline':20}{'AURC':>9}"
          + "".join(f"{f'cov@{b:.0%}':>11}" for b in FALSE_ACCEPT_BUDGETS))
    for row in rows:
        print(f"{row['baseline']:20}{row['aurc']:9.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}']:11.3f}"
                        for b in FALSE_ACCEPT_BUDGETS))

    key = f"coverage_at_{TARGET_BUDGET:.0%}"
    delta = rows[1][key] - rows[0][key]
    outcome = "beats" if delta > 0 else ("matches" if delta == 0 else "loses to")
    print(f"\nB3 {outcome} B2 on coverage at {TARGET_BUDGET:.0%} {TARGET_HARM}: {delta:+.3f}")

    report_root.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_") + ("" if args.evidence == "none"
                                           else f"_{args.evidence}")
    with (report_root / f"b3_vlm_{slug}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (report_root / f"b3_vlm_scores_{slug}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "proposal_id", "verdict", "predicted_class", "verified_class",
            "b2_score", "vlm_p_yes"])
        writer.writeheader()
        for position, index in enumerate(tail):
            record = val[index]
            writer.writerow({
                "proposal_id": record["proposal_id"], "verdict": record["verdict"],
                "predicted_class": record["predicted_class"],
                "verified_class": record["verified_class"],
                "b2_score": round(float(b2_scores[index]), 6),
                "vlm_p_yes": round(float(vlm_scores[position]), 6),
            })
    print(f"\nWrote {report_root}/b3_vlm_{slug}.csv")


def _self_check() -> None:
    from math import log

    # Both branches present: renormalise over yes/no and ignore the rest.
    both = {"top_logprobs": [
        {"token": "Yes", "logprob": log(0.6)},
        {"token": "No", "logprob": log(0.2)},
        {"token": "The", "logprob": log(0.2)},
    ]}
    assert abs(yes_probability(both) - 0.75) < 1e-9, yes_probability(both)

    # Case variants of the same branch add together, not compete.
    variants = {"top_logprobs": [
        {"token": "Yes", "logprob": log(0.4)},
        {"token": "yes", "logprob": log(0.2)},
        {"token": "No", "logprob": log(0.2)},
    ]}
    assert abs(yes_probability(variants) - 0.75) < 1e-9, yes_probability(variants)

    # Only the argmax returned (the 8B arm): must NOT renormalise to 1.0.
    only_yes = {"top_logprobs": [{"token": "Yes", "logprob": log(0.66)}]}
    assert abs(yes_probability(only_yes) - 0.66) < 1e-9, yes_probability(only_yes)
    only_no = {"top_logprobs": [{"token": "No", "logprob": log(0.9)}]}
    assert abs(yes_probability(only_no) - 0.10) < 1e-9, yes_probability(only_no)
    # ... and a confident yes must still outrank a confident no.
    assert yes_probability(only_yes) > yes_probability(only_no)
    # A less confident yes ranks below a more confident one.
    assert yes_probability({"top_logprobs": [{"token": "Yes", "logprob": log(0.51)}]}) \
        < yes_probability(only_yes)

    # Neither branch is an error, not a silent default.
    try:
        yes_probability({"top_logprobs": [{"token": "The", "logprob": log(0.9)}]})
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted a position with no yes/no mass")

    # --- find_answer_position: reasoning-model sequence scanning -----------
    def entry(token: str) -> dict:
        return {"token": token, "top_logprobs": [{"token": token, "logprob": 0.0}]}

    # Old direct-answer shape: the only token IS the answer.
    assert find_answer_position([entry("Yes")]) == 0

    # A reasoning trace: "a yes/no question" at position 2 is a false-positive
    # trap: the real answer, after </think>, must win, not the first match.
    trace = [entry("The"), entry(" is"), entry(" yes"), entry("/no"),
             entry(" question"), entry("</think>"), entry("Yes")]
    assert find_answer_position(trace) == 6, find_answer_position(trace)

    # No yes/no token anywhere -> None, not a crash or a wrong guess.
    assert find_answer_position([entry("The"), entry(" cat")]) is None

    # --- the evidence arms -------------------------------------------------
    names = ["ship", "vehicle", "airplane", "harbor", "bridge"]
    rng = np.random.default_rng(SEED)
    drawn = counterfactual_classes(names, "ship", "vehicle", 5, rng)
    assert len(drawn) == 5
    # Neither the claimed class nor the true one may ever appear, or the
    # control is measuring something other than "retrieval says otherwise".
    assert not ({"ship", "vehicle"} & set(drawn.tolist())), drawn
    try:
        counterfactual_classes(["ship", "vehicle"], "ship", "vehicle", 3, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a class list with no counterfactual left")

    # The three arms must produce three genuinely different prompts, and the
    # ungrounded one must contain no evidence at all.
    true_block = evidence_block(["ship", "ship", "ship"], [0.91, 0.88, 0.85])
    counter_block = evidence_block(["bridge", "harbor", "bridge"], [0.91, 0.88, 0.85])
    none_prompt = build_prompt("ship", None)
    true_prompt = build_prompt("ship", true_block)
    counter_prompt = build_prompt("ship", counter_block)
    assert len({none_prompt, true_prompt, counter_prompt}) == 3
    assert "verified" not in none_prompt and "ship" in none_prompt
    # Same similarities in both grounded arms: only the labels differ, so any
    # score change is attributable to the labels and not to the numbers.
    for score in ("0.91", "0.88", "0.85"):
        assert score in true_prompt and score in counter_prompt
    assert "verified ship" in true_prompt and "verified ship" not in counter_prompt

    print("b3 self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
