"""Diagnose live-vs-reference embedding mismatch.

Compares three ways of producing the pooled embedding for song position T
(pooling the LOOKBACK_SEC window ending at T) and reports where each one's
nearest reference embedding lands:

  A. reference itself        (sanity: should land exactly at T)
  B. one MERT call on the whole 2s window          (chunked-context test)
  C. live-style: MERT on consecutive 0.2s chunks, frames concatenated
     (this is exactly what SongAligner._embed_chunk does)

Usage: python tools/diag_embed.py data/incubus/drive/incubus_drive_cache.npz --file <wav> --t 36 64 130
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ppsync.config import CHUNK_SEC, LOOKBACK_SEC, TARGET_SR
from ppsync.embed import embed_chunk_live, load_model
from ppsync.io import load_audio
from ppsync.preprocess import load_cache
from ppsync.transform import apply_contrastive


def nearest(pooled_norm: np.ndarray, ref_embs: np.ndarray, ref_ts: np.ndarray) -> str:
    sims = ref_embs @ pooled_norm
    i = int(np.argmax(sims))
    return f"best={ref_ts[i]:7.2f}s sim={sims[i]:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("--file", required=True)
    p.add_argument("--t", type=float, nargs="+", default=[36.0, 64.0, 130.0])
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    cache = load_cache(args.cache)
    ref_embs, ref_ts = cache["ref_embs"], cache["ref_timestamps"]
    global_emb = torch.from_numpy(cache["global_emb"])
    layer = int(cache["mert_layer"])

    wav = load_audio(args.file)
    if torch.is_tensor(wav):
        wav = wav.numpy()
    processor, model = load_model(args.device)

    for t_end in args.t:
        s0 = int((t_end - LOOKBACK_SEC) * TARGET_SR)
        s1 = int(t_end * TARGET_SR)
        window = torch.from_numpy(wav[s0:s1])

        # B: full-window single MERT call
        frames_b = embed_chunk_live(window, model, processor, args.device)[layer]
        pooled_b = apply_contrastive(frames_b.mean(dim=0), global_emb).numpy()

        # C: live-style chunked MERT
        chunk_n = int(CHUNK_SEC * TARGET_SR)
        frames_c = []
        for c0 in range(0, len(window), chunk_n):
            chunk = window[c0:c0 + chunk_n]
            if len(chunk) < chunk_n // 2:
                break
            frames_c.append(embed_chunk_live(chunk, model, processor, args.device)[layer])
        frames_c = torch.cat(frames_c, dim=0)
        pooled_c = apply_contrastive(frames_c.mean(dim=0), global_emb).numpy()

        # A: the reference embedding at this timestamp
        ref_i = int(np.argmin(np.abs(ref_ts - t_end)))
        pooled_a = ref_embs[ref_i]

        print(f"\nT={t_end:.1f}s  (window {t_end - LOOKBACK_SEC:.1f}–{t_end:.1f}s)")
        print(f"  A ref-self      : {nearest(pooled_a, ref_embs, ref_ts)}")
        print(f"  B full-window   : {nearest(pooled_b, ref_embs, ref_ts)}")
        print(f"  C live-chunked  : {nearest(pooled_c, ref_embs, ref_ts)}")
        print(f"  cos(B, A)={float(pooled_b @ pooled_a):.4f}   "
              f"cos(C, A)={float(pooled_c @ pooled_a):.4f}   "
              f"cos(C, B)={float(pooled_c @ pooled_b):.4f}")


if __name__ == "__main__":
    main()
