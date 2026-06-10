"""Phase-sensitivity test for sliding-window MERT embeddings.

The chunked live path fails when playback start is not aligned to the 0.2s
chunk grid (offset 64.1 benchmark: 4% tracking vs 100% aligned).  Hypothesis:
embedding the FULL lookback window in one MERT call per update removes chunk
phase from the equation — the embedding becomes a pure function of the last
LOOKBACK_SEC of audio, and the worst-case phase error vs a strided reference
is half the reference stride.

Test: build a reference of full-window embeddings at a fine stride over a
region of the song, then query with windows whose right edges deliberately
fall OFF the reference grid.  Report where the nearest reference lands and the
similarity margin over the rest of the region.

Usage: python tools/diag_phase.py --file <wav> [--lo 30 --hi 50 --stride 0.05]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ppsync.config import LOOKBACK_SEC, MERT_LAYER, TARGET_SR
from ppsync.embed import embed_chunk_live, load_model
from ppsync.io import load_audio
from ppsync.transform import apply_contrastive, fit_global


def window_emb(wav: np.ndarray, t_end: float, model, proc, device: str) -> torch.Tensor:
    """Raw mean-pooled MERT embedding of the window ending at *t_end*."""
    s0 = int((t_end - LOOKBACK_SEC) * TARGET_SR)
    s1 = int(t_end * TARGET_SR)
    frames = embed_chunk_live(torch.from_numpy(wav[s0:s1]), model, proc, device)[MERT_LAYER]
    return frames.mean(dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--lo", type=float, default=30.0, help="Region start (window right edges).")
    p.add_argument("--hi", type=float, default=50.0)
    p.add_argument("--stride", type=float, default=0.05)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    wav = load_audio(args.file)
    if torch.is_tensor(wav):
        wav = wav.numpy()
    proc, model = load_model(args.device)

    edges = np.arange(args.lo, args.hi, args.stride)
    print(f"Building {len(edges)} reference windows ({args.lo}-{args.hi}s, stride {args.stride}s)…")
    raw = torch.stack([window_emb(wav, t, model, proc, args.device) for t in edges])
    global_emb = fit_global(raw)
    ref = apply_contrastive(raw, global_emb).numpy()

    # Query edges intentionally off the reference grid by sub-stride amounts.
    queries = [args.lo + 5.013, args.lo + 9.127, args.lo + 13.061, args.lo + 16.077]
    print(f"\n{'query_t':>9} {'best_t':>8} {'err_ms':>7} {'sim':>7} {'margin':>7}")
    for q in queries:
        q_emb = apply_contrastive(window_emb(wav, q, model, proc, args.device), global_emb).numpy()
        sims = ref @ q_emb
        i = int(np.argmax(sims))
        # margin: best sim minus best sim outside +-0.5s of the true position
        far = np.abs(edges - q) > 0.5
        margin = float(sims[i] - sims[far].max()) if far.any() else float("nan")
        print(f"{q:9.3f} {edges[i]:8.3f} {(edges[i]-q)*1000:7.0f} {sims[i]:7.3f} {margin:7.3f}")


if __name__ == "__main__":
    main()
