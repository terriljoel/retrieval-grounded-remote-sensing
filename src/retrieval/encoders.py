"""Four image encoders, as a dict of loader functions.

Four is not enough to justify a class hierarchy. Each loader returns
(encode, preprocess): `encode` maps a batched float tensor to a float32
numpy array; `preprocess` maps a PIL image to a tensor.

Normalisation is per encoder -- CLIP constants must not be applied to
DINOv2. Crops are written at exactly 224x224, so every transform is
resize-to-224 + normalise with that encoder's own constants; notably no
resize-256-then-centre-crop-224, which would silently discard 12.5% of the
context crop and make the encoders non-comparable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np

CROP_SIZE = 224
REMOTECLIP_REPO = "chendelong/RemoteCLIP"


def _device(prefer_cuda: bool = True) -> str:
    import torch

    return "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"


def _transform(mean, std):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _clip_constants(preprocess):
    """Pull the mean/std out of an open_clip preprocess pipeline."""
    for step in reversed(preprocess.transforms):
        if hasattr(step, "mean") and hasattr(step, "std"):
            return step.mean, step.std
    raise RuntimeError("open_clip preprocess has no Normalize step")


def _load_open_clip(model_name: str, pretrained: str | None, remoteclip: bool, device: str):
    import logging
    import warnings

    import open_clip
    import torch

    # open_clip only *warns* on a QuickGELU mismatch and then builds a model
    # whose activation disagrees with its weights. For the OpenAI control that
    # would silently handicap the very baseline RemoteCLIP is measured against,
    # so treat it as an error.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if remoteclip:
            # RemoteCLIP is not a registered open_clip checkpoint, so it is
            # always built with pretrained=None (an empty architecture) and
            # its real weights are loaded explicitly below. open_clip logs
            # "No pretrained weights loaded ... initialized randomly" for
            # every pretrained=None build via logging.warning() straight to
            # the root logger (open_clip/factory.py) -- NOT via warnings.warn,
            # so the warnings.catch_warnings() above never sees it. That
            # message is indistinguishable in the log from the actual failure
            # this function exists to prevent (a RemoteCLIP encoder silently
            # staying randomly initialised), so it is suppressed here rather
            # than left to be misread -- the real guarantee is the explicit
            # load + verification below, which raises, not warns.
            root_logger = logging.getLogger()
            previous_level = root_logger.level
            root_logger.setLevel(logging.ERROR)
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
            finally:
                root_logger.setLevel(previous_level)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
    mismatch = [str(w.message) for w in caught if "QuickGELU" in str(w.message)]
    if mismatch:
        raise RuntimeError(
            f"{model_name}/{pretrained}: {mismatch[0]} "
            "Use the -quickgelu model name for OpenAI weights."
        )
    if remoteclip:
        from huggingface_hub import hf_hub_download

        # A randomly-initialised RemoteCLIP encoder returns unit-norm vectors
        # and plausible-looking cosine similarities with no error at all --
        # the same silent-failure class as the QuickGELU mismatch above, the
        # degenerate class_margin sentinel, and the missing-logprob branch in
        # run_b3_vlm.py. Every step from here on is wrapped so any failure
        # (network, a corrupted download, a state-dict mismatch) becomes an
        # unambiguous RuntimeError naming RemoteCLIP, never a silent fallback
        # to the empty architecture built above.
        try:
            checkpoint = hf_hub_download(
                REMOTECLIP_REPO, f"RemoteCLIP-{model_name}.pt", cache_dir="checkpoints"
            )
            state_dict = torch.load(checkpoint, map_location="cpu")
        except Exception as error:
            raise RuntimeError(
                f"RemoteCLIP-{model_name}: failed to fetch or read the checkpoint "
                f"({REMOTECLIP_REPO}, cache_dir='checkpoints', resolved relative to "
                f"the current working directory). {type(error).__name__}: {error}"
            ) from error
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"RemoteCLIP-{model_name} state dict mismatch: "
                f"{len(missing)} missing, {len(unexpected)} unexpected. "
                f"First missing: {missing[:3]}. The encoder would otherwise be "
                f"left randomly initialised and silently return plausible-looking "
                f"similarity scores."
            )
    model = model.to(device).eval()
    mean, std = _clip_constants(preprocess)

    @torch.no_grad()
    def encode(batch):
        return model.encode_image(batch.to(device)).float().cpu().numpy()

    return encode, _transform(mean, std)


def _load_dinov2(device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def encode(batch):
        # CLS token is DINOv2's global descriptor for retrieval.
        output = model(pixel_values=batch.to(device)).last_hidden_state[:, 0]
        return output.float().cpu().numpy()

    return encode, _transform(processor.image_mean, processor.image_std)


ENCODERS: dict[str, Callable[[str], tuple]] = {
    "remoteclip_b32": lambda device: _load_open_clip("ViT-B-32", None, True, device),
    "remoteclip_l14": lambda device: _load_open_clip("ViT-L-14", None, True, device),
    # OpenAI weights require QuickGELU; the plain ViT-B-32 config would build
    # the wrong activation and understate the control.
    "openclip_b32": lambda device: _load_open_clip("ViT-B-32-quickgelu", "openai", False, device),
    "dinov2_b14": lambda device: _load_dinov2(device),
}


@lru_cache(maxsize=1)
def load_encoder(key: str, device: str | None = None):
    """Build an encoder. Cached, because the sweep embeds one encoder across
    many streams and reloading ViT-L/14 from disk each time costs minutes.

    maxsize=1 on purpose: holding all four encoders would exceed 6 GB of VRAM,
    so a new key evicts the previous model rather than accumulating.
    """
    if key not in ENCODERS:
        raise KeyError(f"Unknown encoder {key!r}. Known: {sorted(ENCODERS)}")
    return ENCODERS[key](device or _device())


def l2_normalise(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("Zero-norm embedding cannot be L2-normalised")
    return (embeddings / norms).astype(np.float32)
