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
song_id           str scalar   song title from the manifest (e.g. "Drive")
artist            str scalar   artist from the manifest (e.g. "Incubus")
song_slug         str scalar   artist_title filename slug (e.g. "incubus_drive")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import CHUNK_SEC, LOOKBACK_SEC, MERT_FP16, MERT_LAYER, STRIDE_SEC, TARGET_SR
from .embed import embed_audio, load_model, prep_inputs
from .io import finalize_slide_stops, load_audio, load_manifest, load_song_meta
from .transform import apply_contrastive, fit_global
from .windows import pool_slide_embeddings, strided_window_embeddings


# ---------------------------------------------------------------------------
# Sliding-window reference embeddings
# ---------------------------------------------------------------------------

def sliding_window_embeddings(
    wav: torch.Tensor,
    model,
    processor,
    device: str,
    lookback_sec: float = LOOKBACK_SEC,
    stride_sec: float = STRIDE_SEC,
    mert_layer: int = MERT_LAYER,
    batch_size: int = 16,
    show_progress: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Embed each [t - lookback_sec, t] window in its own MERT forward pass
    (batched) and mean-pool the frames.

    Unlike chunked embedding, the result is a pure function of the window's
    audio — independent of any chunk grid — so live queries whose start time
    is not phase-aligned with the reference still match (the chunked pipeline
    drops from 100% to 4% tracking on a 0.1s phase shift).

    Returns:
        win_embs:   [N, D] raw mean-pooled window embeddings
        timestamps: [N] right-edge time of each window (seconds)
    """
    from tqdm import tqdm

    sr = TARGET_SR
    win = int(lookback_sec * sr)
    duration = wav.shape[0] / sr
    edges = np.arange(lookback_sec, duration + 1e-9, stride_sec)

    embs: list[torch.Tensor] = []
    batches = range(0, len(edges), batch_size)
    if show_progress:
        batches = tqdm(batches, unit="batch", desc="Sliding windows")
    with torch.no_grad():
        for b0 in batches:
            windows = []
            for t in edges[b0:b0 + batch_size]:
                s1 = min(int(t * sr), wav.shape[0])
                windows.append(wav[s1 - win:s1].numpy())
            inputs = prep_inputs(
                processor(windows, sampling_rate=sr, return_tensors="pt"), model
            )
            out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[mert_layer]  # [B, T, D]
            embs.append(h.mean(dim=1).float().cpu())

    return torch.cat(embs, dim=0), edges


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
    embed_chunk_sec: float = CHUNK_SEC,
    embed_mode: str = "sliding",
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
    meta = load_song_meta(manifest_path)
    audio_path, slides = load_manifest(manifest_path)
    artist_str = meta["artist"] or "(unknown artist)"
    print(f"  Song: {artist_str} — {meta['song_id']}  [slug: {meta['slug']}]")
    if not meta["artist"]:
        print("  WARNING: manifest has no 'artist' field — add one so cache/log "
              "filenames identify the song unambiguously.")
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

    # Reference embeddings MUST come from the same computation the live path
    # uses: MERT frames depend on their attention context, so embeddings from
    # 30s chunks live in a different distribution than live ones and cosine
    # matching across the two fails entirely.
    processor, model = load_model(device, truncate_after_layer=mert_layer)
    frame_rate = float(0)
    if embed_mode == "sliding":
        # One forward pass per [t - lookback, t] window — phase-independent,
        # matching the live path which re-embeds its full lookback each chunk.
        print(f"\nRunning MERT on {device}  (layer {mert_layer}, sliding "
              f"{lookback_sec}s windows, stride {stride_sec}s)…")
        raw_win_embs, ref_timestamps = sliding_window_embeddings(
            wav, model, processor, device,
            lookback_sec=lookback_sec, stride_sec=stride_sec,
            mert_layer=mert_layer, show_progress=show_progress,
        )
        print(f"  Reference windows: {raw_win_embs.shape[0]:,}")
    else:
        print(f"\nRunning MERT on {device}  (layer {mert_layer}, chunk={embed_chunk_sec}s)…")
        hidden = embed_audio(
            wav, model, processor, device,
            chunk_sec=embed_chunk_sec, show_progress=show_progress,
        )
        # hidden: [L+1, T, D]
        frames = hidden[mert_layer]  # [T, D]
        # Short chunks lose conv edge frames (0.2s -> 14 frames = 70fps, not
        # 75); derive the effective rate so timestamps stay true to song time.
        frame_rate = frames.shape[0] / song_duration
        print(f"  Frames: {frames.shape[0]:,}  Dim: {frames.shape[1]}  "
              f"({frame_rate:.2f} fps effective)")

        print(f"\nBuilding dense reference embeddings "
              f"(lookback={lookback_sec}s, stride={stride_sec}s)…")
        raw_win_embs, ref_timestamps = strided_window_embeddings(
            frames, lookback_sec=lookback_sec, stride_sec=stride_sec, fps=frame_rate
        )
    print(f"  Reference windows: {raw_win_embs.shape[0]:,}")

    global_emb = fit_global(raw_win_embs)  # [D]

    ref_embs = apply_contrastive(raw_win_embs, global_emb)  # [N_ref, D]

    print("Building slide prototypes…")
    slide_protos_list: list[torch.Tensor] = []
    slide_t_refs = np.array([s["t_ref"] for s in slides], dtype=np.float64)
    slide_t_stops = np.array([s["t_stop"] for s in slides], dtype=np.float64)
    slide_ids = np.array([s["slide_id"] for s in slides], dtype=object)
    slide_pp_indices = np.array(
        [s.get("pp_slide_index", i) for i, s in enumerate(slides)], dtype=np.int32
    )
    pp_uuid = meta["pp_uuid"]

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
        "frame_rate": np.float32(frame_rate),
        "embed_chunk_sec": np.float32(embed_chunk_sec),
        "embed_mode": np.array(embed_mode),
        "slide_pp_indices": slide_pp_indices,
        "pp_uuid": np.array(pp_uuid),
        "mert_fp16": np.bool_(MERT_FP16 and device != "cpu"),
        "song_id": np.array(meta["song_id"]),
        "artist": np.array(meta["artist"]),
        "song_slug": np.array(meta["slug"]),
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
