"""Offline preprocessing: build reference embedding cache for one song.

Saved cache (.npz) layout
--------------------------
ref_embs          [N_ref, D]   contrastive L2-normalized dense embeddings
ref_timestamps    [N_ref]      right-edge time (seconds) for each window
slide_protos      [N_slides, D] per-slide prototype embeddings (same transform)
slide_t_refs      [N_slides]   slide start times from JSON
slide_t_stops     [N_slides]   slide stop times (= next slide start / song end)
slide_ids         object array  slide ID strings
global_emb        [D]          song-level mean (needed to normalize live frames)
hmm_A             [N_slides, N_slides]  HMM transition matrix
hmm_pi            [N_slides]   initial state distribution (uniform)
song_duration     scalar       total audio duration in seconds
lookback_sec      scalar       window lookback used during preprocessing
stride_sec        scalar       window stride used during preprocessing
mert_layer        scalar int   which MERT layer was used
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import LOOKBACK_SEC, MERT_FRAME_RATE, MERT_LAYER, STRIDE_SEC, TARGET_SR
from .embed import embed_audio, load_model
from .io import finalize_slide_stops, load_audio, load_manifest
from .transform import apply_contrastive, fit_global
from .windows import pool_slide_embeddings, strided_window_embeddings


# ---------------------------------------------------------------------------
# HMM transition matrix
# ---------------------------------------------------------------------------

def build_hmm_transition(
    slide_t_refs: np.ndarray,
    slide_t_stops: np.ndarray,
    stride_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an HMM left-to-right transition matrix from slide durations.

    The update interval is *stride_sec* (the same as the reference stride, so
    the HMM advances one step per reference embedding).

    Transition probabilities:
        A[i][i]   = self-loop   = max(0, 1 - stride_sec / duration_i)
        A[i][i+1] = forward     = 1 - A[i][i]   (capped so it's never > 1)
        A[N-1][N-1] = 1.0       (last slide — absorbing)

    Args:
        slide_t_refs:  [N] slide start times
        slide_t_stops: [N] slide stop times
        stride_sec:    HMM step interval in seconds

    Returns:
        A:  [N, N] transition matrix (row = from, col = to)
        pi: [N]    uniform initial distribution
    """
    N = len(slide_t_refs)
    durations = slide_t_stops - slide_t_refs  # [N]
    durations = np.maximum(durations, stride_sec)  # avoid division by zero

    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        p_forward = min(1.0, stride_sec / durations[i])
        p_forward = max(0.001, p_forward)  # small minimum to allow progress
        p_self = 1.0 - p_forward
        A[i, i] = p_self
        if i + 1 < N:
            A[i, i + 1] = p_forward
        else:
            A[i, i] = 1.0  # last state absorbs

    pi = np.ones(N, dtype=np.float64) / N  # uniform prior
    return A, pi


# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess_song(
    manifest_path: Path,
    output_path: Path,
    lookback_sec: float = LOOKBACK_SEC,
    stride_sec: float = STRIDE_SEC,
    mert_layer: int = MERT_LAYER,
    device: str | None = None,
    show_progress: bool = True,
) -> dict:
    """
    Run offline preprocessing for one song and save the embedding cache.

    Steps:
        1. Load manifest + audio
        2. Run MERT over the full audio (chunked)
        3. Build dense strided window embeddings (stride_sec, lookback_sec)
        4. Compute global (song-level) embedding
        5. Apply contrastive normalization (subtract global, L2-normalize)
        6. Pool per-slide prototypes with the same normalization
        7. Build HMM transition matrix from slide durations
        8. Save everything to output_path (.npz)

    Returns the saved dict (for in-memory reuse by eval pipeline).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading manifest: {manifest_path}")
    audio_path, slides = load_manifest(manifest_path)
    print(f"  {len(slides)} slides, audio: {audio_path}")

    print(f"Loading audio…")
    wav = load_audio(audio_path)
    song_duration = float(wav.shape[0]) / TARGET_SR
    finalize_slide_stops(slides, song_duration)
    print(f"  Duration: {song_duration:.2f}s  ({wav.shape[0]:,} samples @ {TARGET_SR} Hz)")

    for s in slides:
        print(
            f"    [{s['slide_id']:20s}]  "
            f"{s['t_ref']:6.2f}s – {s['t_stop']:6.2f}s  "
            f"({s['t_stop'] - s['t_ref']:.2f}s)"
        )

    print(f"\nRunning MERT on {device}  (layer {mert_layer})…")
    processor, model = load_model(device)
    hidden = embed_audio(wav, model, processor, device, show_progress=show_progress)
    # hidden: [L+1, T, D]
    frames = hidden[mert_layer]  # [T, D]
    print(f"  Frames: {frames.shape[0]:,}  Dim: {frames.shape[1]}")

    print(f"\nBuilding dense reference embeddings (lookback={lookback_sec}s, stride={stride_sec}s)…")
    raw_win_embs, ref_timestamps = strided_window_embeddings(
        frames, lookback_sec=lookback_sec, stride_sec=stride_sec, fps=MERT_FRAME_RATE
    )
    print(f"  Reference windows: {raw_win_embs.shape[0]:,}")

    global_emb = fit_global(raw_win_embs)  # [D]

    ref_embs = apply_contrastive(raw_win_embs, global_emb)  # [N_ref, D]

    print("Building slide prototypes…")
    slide_protos_list: list[torch.Tensor] = []
    slide_t_refs = np.array([s["t_ref"] for s in slides], dtype=np.float64)
    slide_t_stops = np.array([s["t_stop"] for s in slides], dtype=np.float64)
    slide_ids = np.array([s["slide_id"] for s in slides], dtype=object)

    for s in slides:
        raw_proto = pool_slide_embeddings(
            raw_win_embs, ref_timestamps, s["t_ref"], s["t_stop"]
        )  # [D] raw mean
        proto = apply_contrastive(raw_proto, global_emb)  # [D] normalized
        slide_protos_list.append(proto)
    slide_protos = torch.stack(slide_protos_list)  # [N_slides, D]

    print("Building HMM transition matrix…")
    hmm_A, hmm_pi = build_hmm_transition(slide_t_refs, slide_t_stops, stride_sec)

    payload = {
        "ref_embs": ref_embs.numpy().astype(np.float32),
        "ref_timestamps": ref_timestamps.astype(np.float32),
        "slide_protos": slide_protos.numpy().astype(np.float32),
        "slide_t_refs": slide_t_refs.astype(np.float32),
        "slide_t_stops": slide_t_stops.astype(np.float32),
        "slide_ids": slide_ids,
        "global_emb": global_emb.numpy().astype(np.float32),
        "hmm_A": hmm_A.astype(np.float64),
        "hmm_pi": hmm_pi.astype(np.float64),
        "song_duration": np.float32(song_duration),
        "lookback_sec": np.float32(lookback_sec),
        "stride_sec": np.float32(stride_sec),
        "mert_layer": np.int32(mert_layer),
    }
    np.savez(str(output_path), **payload)
    print(f"\nCache saved to: {output_path}")
    return payload


def load_cache(cache_path: Path) -> dict:
    """
    Load a preprocessing cache produced by preprocess_song.

    All numpy arrays are returned as-is; object arrays (slide_ids) are
    converted to plain Python lists of strings.
    """
    raw = np.load(str(cache_path), allow_pickle=True)
    cache = {k: raw[k] for k in raw.files}
    cache["slide_ids"] = cache["slide_ids"].tolist()
    return cache
