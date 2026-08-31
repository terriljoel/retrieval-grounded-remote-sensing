"""Settle, empirically, what the NIM endpoint will give us before B3 is built on it.

Three questions, in order:

  1. Which vision models does this endpoint actually serve?
  2. Does a VLM chat completion accept `logprobs` / `top_logprobs`, and does the
     response carry them? The whole yes/no-logit scoring design depends on this,
     because a risk-coverage curve needs a continuous score and a generated
     "yes" is binary.
  3. If not, does the model at least produce a usable elicited confidence, and
     is sampling at temperature > 0 varied enough for self-consistency to
     produce a gradient rather than a constant?

Run it before choosing a scoring method. It costs a handful of calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.retrieval.nim import (
    KEY_VARIABLE, api_key, call, encode_image, list_models, model_name,
    request_body, temperature as env_temperature, usage_report,
)
from src.utils.paths import find_project_root

VISION_HINTS = ("vl", "vision", "gemma-3", "kosmos", "nemotron-nano-12b",
                "nemotron-nano-vl", "cosmos", "maverick", "scout", "paligemma")
# Last-resort ordering when NIM_MODEL is unset. **No Meta models**: their
# multimodal licence excludes the EU, so llama-3.2-*-vision cannot be used from
# Germany at all -- a hard legal constraint, not a preference. Listing them here
# would let a missing NIM_MODEL silently select an unusable model.
PREFERRED = (
    "nvidia/nemotron-nano-12b-v2-vl",
    "google/gemma-3-12b-it",
    "microsoft/phi-3-vision-128k-instruct",
)
EU_BLOCKED_PREFIXES = ("meta/",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Override the model to probe.")
    parser.add_argument("--samples", type=int, default=5,
                        help="Sampling draws for the self-consistency probe.")
    return parser.parse_args()


def probe_logprobs(model: str, prompt: str, image_url: str, cache_root: Path) -> dict:
    """Ask for logprobs two ways and record exactly what came back."""
    findings = {}
    for label, options in (
        ("logprobs+top_logprobs", {"logprobs": True, "top_logprobs": 5, "max_tokens": 1}),
        ("logprobs_only", {"logprobs": True, "max_tokens": 1}),
    ):
        body = request_body(model, prompt, image_url, temperature=0.0, **options)
        try:
            payload = call(body, cache_root=cache_root)
        except RuntimeError as error:
            findings[label] = {"accepted": False, "error": str(error)[:300]}
            continue
        choice = (payload.get("choices") or [{}])[0]
        returned = choice.get("logprobs")
        findings[label] = {
            "accepted": True,
            "logprobs_in_response": returned is not None,
            "sample": json.dumps(returned)[:400] if returned else None,
            "content": (choice.get("message") or {}).get("content"),
        }
    return findings


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    cache_root = root / "data" / "nim_cache"

    print("=== 1. vision models offered ===")
    try:
        models = list_models()
    except Exception as error:
        print(f"  could not list models: {error}")
        models = []
    vision = [m for m in models if any(h in m.lower() for h in VISION_HINTS)]
    for name in vision:
        blocked = name.startswith(EU_BLOCKED_PREFIXES)
        print(f"  {name}{'   [EU-BLOCKED: Meta multimodal licence]' if blocked else ''}")
    vision = [m for m in vision if not m.startswith(EU_BLOCKED_PREFIXES)]
    if not any("qwen" in m.lower() for m in models):
        print("  NOTE: no Qwen model on this endpoint at all "
              f"({len(models)} models listed).")

    # NIM_MODEL is the source of truth; PREFERRED is only a last resort.
    model = args.model or model_name() or next(
        (m for m in PREFERRED if m in models), vision[0] if vision else None)
    if model is None:
        print("\nNo vision model available; stopping.")
        return
    print(f"\nProbing: {model}")

    try:
        api_key()
    except RuntimeError as error:
        print(f"\n{error}\n")
        print("Model listing worked without a key, but the probes below need one.")
        return

    # A trivially unambiguous image, so a wrong answer means a broken call
    # rather than a hard question.
    image = Image.new("RGB", (224, 224), (200, 30, 30))
    image_url = encode_image(image)
    prompt = "Is this image entirely red? Answer only yes or no."

    print("\n=== 2. logprobs support ===")
    findings = probe_logprobs(model, prompt, image_url, cache_root)
    for label, result in findings.items():
        if not result["accepted"]:
            print(f"  {label}: REJECTED -- {result['error']}")
        else:
            print(f"  {label}: accepted, logprobs in response: "
                  f"{result['logprobs_in_response']}")
            if result["sample"]:
                print(f"    {result['sample']}")

    usable = any(r.get("logprobs_in_response") for r in findings.values())
    print(f"\n  => continuous scoring by token logprob is "
          f"{'AVAILABLE' if usable else 'NOT AVAILABLE'}")

    # Self-consistency at N=10 re-sends the same image ten times. If the
    # endpoint honours `n`, it is one image upload and ten completions instead
    # -- roughly a 10x cost cut, since the image dominates the prompt.
    print("\n=== 2b. does `n` (multiple completions) work? ===")
    try:
        payload = call(request_body(model, prompt, image_url,
                                    temperature=env_temperature(), max_tokens=4, n=4),
                       cache_root=cache_root)
        returned = len(payload.get("choices") or [])
        usage = payload.get("usage") or {}
        print(f"  accepted; returned {returned} choices, "
              f"prompt_tokens={usage.get('prompt_tokens')}")
        print(f"  => {'USABLE: batch the samples, ~10x cheaper' if returned > 1 else 'ignored, n=1 returned; pay per sample'}")
    except RuntimeError as error:
        print(f"  REJECTED -- {str(error)[:200]}")
        print("  => pay per sample")

    print("\n=== 2c. per-call token cost ===")
    try:
        payload = call(request_body(model, prompt, image_url,
                                    temperature=env_temperature(), max_tokens=4, seed=0),
                       cache_root=cache_root)
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        print(f"  one yes/no call on a 224x224 crop: {prompt_tokens} prompt + "
              f"{usage.get('completion_tokens')} completion tokens")
        print(f"  at N=10 self-consistency that is ~{prompt_tokens * 10:,} tokens per crop")
    except RuntimeError as error:
        print(f"  could not measure: {str(error)[:200]}")

    if not usable:
        print("\n=== 3. fallbacks ===")
        numeric = request_body(
            model,
            "Is this image entirely red? Answer with only a number from 0 to 100 "
            "giving your confidence that the answer is yes.",
            image_url, temperature=0.0, max_tokens=8,
        )
        payload = call(numeric, cache_root=cache_root)
        content = (payload["choices"][0].get("message") or {}).get("content")
        print(f"  elicited numeric confidence: {content!r}")

        answers = []
        for index in range(args.samples):
            body = request_body(model, prompt, image_url,
                                temperature=env_temperature(), max_tokens=4, seed=index)
            payload = call(body, cache_root=cache_root)
            answers.append((payload["choices"][0].get("message") or {}).get("content"))
        print(f"  self-consistency draws (temperature {env_temperature()}): {answers}")
        print(f"  distinct answers: {len(set(answers))} of {len(answers)} "
              f"-- {'varies, usable as a rate' if len(set(answers)) > 1 else 'constant, gives no gradient'}")

    print(f"\n{usage_report()}")
    print(f"cache: {cache_root}")


if __name__ == "__main__":
    main()
