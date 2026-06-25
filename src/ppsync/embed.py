"""MERT model loading and chunked full-song inference."""

from __future__ import annotations

import math

import torch

from .config import MERT_FP16, MERT_LAYER, MODEL_ID, TARGET_SR

EMBED_CHUNK_SEC = 30.0  # audio processed per forward pass (keeps GPU memory bounded)


def load_model(
    device: str | None = None,
    truncate_after_layer: int | None = MERT_LAYER,
    fp16: bool = MERT_FP16,
) -> tuple:
    """
    Download (or load from cache) MERT and its feature extractor.

    *truncate_after_layer* drops transformer layers beyond the extraction
    layer — hidden_states[k] only depends on layers 1..k, so the output is
    bit-identical while skipping the dead compute (layers 8-12 for layer 7,
    ~40% of the transformer).  *fp16* halves precision for MPS speed;
    reference cache and live inference must use the SAME precision.

    Returns:
        (processor, model)  — model on *device*, in eval mode
    """
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    if truncate_after_layer is not None and 1 <= truncate_after_layer < len(model.encoder.layers):
        model.encoder.layers = model.encoder.layers[:truncate_after_layer]
    if fp16 and device != "cpu":  # fp16 inference is slow/unstable on CPU
        model = model.half()
    model = model.to(device).eval()
    return processor, model


def prep_inputs(inputs, model):
    """Match processor output dtype/device to the model (fp16 support)."""
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    out = {}
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            v = v.to(device=device, dtype=dtype)
        else:
            v = v.to(device=device)
        out[k] = v
    return out


def embed_audio(
    waveform: torch.Tensor,
    model,
    processor,
    device: str,
    chunk_sec: float = EMBED_CHUNK_SEC,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Run MERT over a waveform in fixed-size chunks and concatenate hidden states.

    Chunking keeps memory bounded on long songs.  Chunks are processed
    independently; hidden states are concatenated along the time axis.

    Args:
        waveform:      1-D float tensor at TARGET_SR
        model:         MERT AutoModel (on device, eval mode)
        processor:     Wav2Vec2FeatureExtractor
        device:        torch device string
        chunk_sec:     seconds of audio per forward pass
        show_progress: display tqdm bar

    Returns:
        [num_layers + 1, total_frames, hidden_dim]
    """
    from tqdm import tqdm

    chunk_samples = int(chunk_sec * TARGET_SR)
    total_samples = waveform.shape[0]
    n_chunks = math.ceil(total_samples / chunk_samples)
    duration_sec = total_samples / TARGET_SR

    all_hidden: list[torch.Tensor] = []
    bar = tqdm(
        total=duration_sec,
        unit="s",
        disable=not show_progress,
        desc="Embedding",
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    )
    with bar:
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[start:end]

            inputs = prep_inputs(
                processor(chunk.numpy(), sampling_rate=TARGET_SR, return_tensors="pt"),
                model,
            )

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (L+1) tensors, each [1, T, D]
            hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
            all_hidden.append(hidden.float().cpu())
            bar.update((end - start) / TARGET_SR)

    return torch.cat(all_hidden, dim=1)  # [L+1, total_T, D]


def embed_chunk_live(
    waveform_chunk: torch.Tensor,
    model,
    processor,
    device: str,
    mert_layer: int | None = None,
) -> torch.Tensor:
    """
    Run MERT on a single short chunk for live inference.

    Args:
        waveform_chunk: 1-D float tensor at TARGET_SR (e.g. 200ms = 4800 samples)
        mert_layer:     if given, return only this layer's frames — the other
                        hidden states are dropped on-device before the fp32 cast
                        and device->CPU copy, so only one tensor is moved
                        (the live loop only ever uses a single layer).  None
                        returns the full stack (diagnostics).

    Returns:
        [num_layers + 1, T_chunk, hidden_dim]  when mert_layer is None
        [T_chunk, hidden_dim]                  when mert_layer is given
    """
    inputs = prep_inputs(
        processor(waveform_chunk.numpy(), sampling_rate=TARGET_SR, return_tensors="pt"),
        model,
    )

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if mert_layer is not None:
        return out.hidden_states[mert_layer].squeeze(0).float().cpu()
    return torch.stack(out.hidden_states, dim=0).squeeze(1).float().cpu()
