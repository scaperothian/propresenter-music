"""MERT model loading and chunked full-song inference."""

from __future__ import annotations

import math

import torch

from .config import MODEL_ID, TARGET_SR

EMBED_CHUNK_SEC = 30.0  # audio processed per forward pass (keeps GPU memory bounded)


def load_model(device: str | None = None) -> tuple:
    """
    Download (or load from cache) MERT and its feature extractor.

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
    model = (
        AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        .to(device)
        .eval()
    )
    return processor, model


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

            inputs = processor(
                chunk.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (L+1) tensors, each [1, T, D]
            hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
            all_hidden.append(hidden.cpu())
            bar.update((end - start) / TARGET_SR)

    return torch.cat(all_hidden, dim=1)  # [L+1, total_T, D]


def embed_chunk_live(
    waveform_chunk: torch.Tensor,
    model,
    processor,
    device: str,
) -> torch.Tensor:
    """
    Run MERT on a single short chunk for live inference.

    Args:
        waveform_chunk: 1-D float tensor at TARGET_SR (e.g. 200ms = 4800 samples)

    Returns:
        [num_layers + 1, T_chunk, hidden_dim]
    """
    inputs = processor(
        waveform_chunk.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    return torch.stack(out.hidden_states, dim=0).squeeze(1).cpu()
