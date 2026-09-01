"""NVIDIA NIM client: hosted VLM calls with on-disk caching and token accounting.

The key comes from the `NIM_API_KEY` environment variable and nowhere else --
never a config file, never a committed default. `.env.example` documents the
name; `.env` is gitignored and is never read by this code, so a key can only
reach the process through the environment.

Every response is cached on disk under the SHA-256 of the exact request body,
which includes the base64 image and the prompt. That makes the proposal hash
and the prompt hash both part of the key by construction, and it makes reruns
free -- important when the same 300-crop escalated band is scored repeatedly
while a report is being written.

Token counts from every call are accumulated in `USAGE` so a run can report
what it cost.

stdlib only. `urllib` does what an SDK would here, and an OpenAI-compatible
POST is not worth a dependency.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
KEY_VARIABLE = "NIM_API_KEY"
# Model and sampling live in the environment so switching arms is a variable
# change, not a code edit. Nothing downstream may hardcode a model id.
DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLES = 10
MAX_ATTEMPTS = 6
BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

USAGE: Counter = Counter()      # prompt_tokens / completion_tokens / total_tokens / calls / cached


def api_key() -> str:
    """The NIM key, or a loud failure explaining exactly how to supply one."""
    key = os.environ.get(KEY_VARIABLE, "").strip()
    if not key:
        raise RuntimeError(
            f"{KEY_VARIABLE} is not set.\n"
            f"  The VLM baseline needs a hosted NIM key. Get one at "
            f"https://build.nvidia.com (any model page -> 'Get API Key'), then:\n"
            f"    export {KEY_VARIABLE}='nvapi-...'\n"
            f"  or copy .env.example to .env, fill it in, and "
            f"`set -a; . ./.env; set +a`.\n"
            f"  The key is read from the environment only and is never written "
            f"to disk by this code."
        )
    return key


def base_url() -> str:
    return os.environ.get("NIM_BASE_URL", "").strip() or DEFAULT_BASE_URL


def model_name() -> str:
    return os.environ.get("NIM_MODEL", "").strip() or DEFAULT_MODEL


def temperature() -> float:
    return float(os.environ.get("NIM_TEMPERATURE", "").strip() or DEFAULT_TEMPERATURE)


def samples() -> int:
    return int(os.environ.get("NIM_SAMPLES", "").strip() or DEFAULT_SAMPLES)


def encode_image(image, quality: int = 90) -> str:
    """A PIL image as a base64 data URL."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def request_body(model: str, prompt: str, image_data_url: str, **options) -> dict:
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        **options,
    }


def cache_key(body: dict) -> str:
    """SHA-256 of the canonical request. Covers image bytes and prompt alike."""
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _post(body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url()}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def call(
    body: dict,
    cache_root: Path | None = None,
    timeout: float = 120.0,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict:
    """POST a chat completion, with disk cache and backoff on rate limits.

    Retries only on transient statuses. A 401 or a 400 is a mistake that will
    repeat identically, so it is raised immediately rather than slept over.
    """
    path = None
    if cache_root is not None:
        path = Path(cache_root) / f"{cache_key(body)}.json"
        if path.is_file():
            USAGE["cached"] += 1
            return json.loads(path.read_text())

    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            payload = _post(body, timeout)
            break
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS:
                detail = error.read().decode(errors="replace")[:400]
                raise RuntimeError(f"NIM {error.code}: {detail}") from error
            last = error
            # Honour Retry-After when the server sends one; it knows better
            # than an exponential guess.
            after = error.headers.get("Retry-After") if error.headers else None
            delay = float(after) if after and after.isdigit() else min(
                BACKOFF_BASE ** attempt, BACKOFF_CAP
            )
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            delay = min(BACKOFF_BASE ** attempt, BACKOFF_CAP)
        if attempt == max_attempts - 1:
            raise RuntimeError(f"NIM failed after {max_attempts} attempts: {last}")
        time.sleep(delay)

    usage = payload.get("usage") or {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        USAGE[field] += int(usage.get(field) or 0)
    USAGE["calls"] += 1

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return payload


def usage_report() -> str:
    return (
        f"{USAGE['calls']} API calls ({USAGE['cached']} served from cache), "
        f"{USAGE['prompt_tokens']} prompt + {USAGE['completion_tokens']} completion "
        f"= {USAGE['total_tokens']} tokens"
    )


def list_models(timeout: float = 30.0) -> list[str]:
    """Model ids the endpoint advertises. Does not require a key."""
    with urllib.request.urlopen(f"{base_url()}/models", timeout=timeout) as response:
        return sorted(item["id"] for item in json.loads(response.read().decode())["data"])


def _self_check() -> None:
    import tempfile

    from PIL import Image

    # The key error must name the variable and the fix, not just fail.
    saved = os.environ.pop(KEY_VARIABLE, None)
    try:
        api_key()
    except RuntimeError as error:
        assert KEY_VARIABLE in str(error) and "build.nvidia.com" in str(error)
    else:
        raise AssertionError("missing key was accepted")
    finally:
        if saved is not None:
            os.environ[KEY_VARIABLE] = saved

    # A whitespace-only key is missing, not present.
    os.environ[KEY_VARIABLE] = "   "
    try:
        api_key()
    except RuntimeError:
        pass
    else:
        raise AssertionError("blank key was accepted")
    os.environ.pop(KEY_VARIABLE)
    if saved is not None:
        os.environ[KEY_VARIABLE] = saved

    # Model and sampling knobs come from the environment, with defaults.
    for variable, getter, override, expected in (
        ("NIM_MODEL", model_name, "some/other-vlm", "some/other-vlm"),
        ("NIM_TEMPERATURE", temperature, "0.3", 0.3),
        ("NIM_SAMPLES", samples, "5", 5),
    ):
        previous = os.environ.get(variable)
        os.environ[variable] = override
        assert getter() == expected, (variable, getter())
        os.environ[variable] = "   "          # blank must fall back, not crash
        getter()
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous

    # Cache key must change with the image and with the prompt, separately.
    red = Image.new("RGB", (8, 8), (255, 0, 0))
    blue = Image.new("RGB", (8, 8), (0, 0, 255))
    a = request_body("m", "is this a ship?", encode_image(red))
    b = request_body("m", "is this a ship?", encode_image(blue))
    c = request_body("m", "is this a harbor?", encode_image(red))
    assert cache_key(a) != cache_key(b), "image change did not change the key"
    assert cache_key(a) != cache_key(c), "prompt change did not change the key"
    assert cache_key(a) == cache_key(request_body("m", "is this a ship?", encode_image(red)))
    # Options are part of the key too: logprobs on and off are different calls.
    assert cache_key(a) != cache_key(
        request_body("m", "is this a ship?", encode_image(red), logprobs=True))

    # A cache hit must not need a key or the network.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{cache_key(a)}.json"
        path.write_text(json.dumps({"choices": [{"message": {"content": "yes"}}]}))
        before = USAGE["cached"]
        got = call(a, cache_root=Path(directory))
        assert got["choices"][0]["message"]["content"] == "yes"
        assert USAGE["cached"] == before + 1

    print("nim self-check OK (no network calls made)")


if __name__ == "__main__":
    _self_check()
