"""Per-chunk latency benchmark: does the aligner keep up with real time?

The real-time loop has a hard budget of one chunk duration (CHUNK_SEC, 200ms):
each audio chunk must be fully processed before the next arrives, or capture
falls behind and the mic queue drains into ever-larger combined chunks.  This
tool measures how much of that budget the pipeline actually spends.

Unlike tools/benchmark.py (which scores alignment ACCURACY and reports latency
as a side metric), this tool focuses on latency alone: it drives a song through
the real process_chunk() path, discards a warm-up window (the first chunk pays
MERT graph compilation / lazy allocation), and reports the full percentile
distribution of end-to-end per-chunk time plus an isolated measurement of the
dominant stage — the MERT forward pass.

Breakdown
---------
    embed       embed_chunk_live() on one full lookback window, measured in
                isolation (the single MERT forward + layer select + CPU copy)
    end-to-end  the whole process_chunk(): embed + contrastive normalize +
                coarse cosine + sequence matcher + HMM + trigger
    rest        end-to-end - embed  (matcher + HMM + trigger + bookkeeping)

The headline number is end-to-end p95 vs the CHUNK_SEC budget: if p95 is well
under budget the loop has headroom; rt_ok% is the fraction of chunks that beat
the budget outright.

Usage:
    python tools/latency_benchmark.py data/incubus/drive/incubus_drive_cache.npz \
        --file <song>.wav [--matcher dtw|rigid] [--device mps] \
        [--measure-chunks 300] [--warmup-chunks 60] [--embed-iters 200] \
        [--json-out /tmp/latency.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from ppsync.aligner import SongAligner
from ppsync.audio_capture import FileCapture
from ppsync.config import CHUNK_SEC, DTW_STEP_PENALTY, MATCHER
from ppsync.embed import embed_chunk_live, load_model


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pcts(samples: list[float]) -> dict:
    """Percentile summary of a latency sample (milliseconds)."""
    if not samples:
        return {k: float("nan") for k in
                ("mean", "p50", "p90", "p95", "p99", "max", "n")}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "n": int(a.size),
    }


def measure(
    aligner: SongAligner,
    audio_path: Path,
    warmup_chunks: int,
    measure_chunks: int,
    embed_iters: int,
) -> dict:
    """Drive *audio_path* through the aligner and time the real-time loop."""
    aligner.reset()
    source = FileCapture(audio_path=audio_path, realtime=False, start_offset_sec=0.0)

    e2e_ms: list[float] = []          # full process_chunk(), post-warmup
    scored = 0
    snapshot_ring: np.ndarray | None = None  # one full lookback window for embed timing

    bar = tqdm(source, total=warmup_chunks + measure_chunks, unit="chunk",
               leave=False, desc="latency", dynamic_ncols=True,
               disable=not sys.stderr.isatty())
    for chunk, song_t in bar:
        frame = aligner.process_chunk(chunk, chunk_wall_t=song_t)
        if frame.get("status"):  # buffering / silence — pipeline not fully active
            continue
        # The audio ring is full and representative once we get scored frames;
        # grab one window for the isolated embed measurement.
        if snapshot_ring is None and len(aligner._audio_ring) == aligner._lookback_samples:
            snapshot_ring = aligner._audio_ring.copy()
        scored += 1
        if scored <= warmup_chunks:
            continue  # discard warm-up (first forward pays graph compile)
        e2e_ms.append(frame["processing_ms"])
        if scored >= warmup_chunks + measure_chunks:
            bar.close()
            break

    # Isolated MERT forward: the dominant stage, timed on a representative
    # full lookback window so it is comparable to the live path exactly.
    embed_ms: list[float] = []
    if snapshot_ring is not None:
        ring = torch.from_numpy(snapshot_ring)
        for _ in range(embed_iters):
            t0 = perf_counter()
            aligner._embed_chunk(ring)
            embed_ms.append((perf_counter() - t0) * 1000.0)

    e2e = _pcts(e2e_ms)
    embed = _pcts(embed_ms)
    budget_ms = CHUNK_SEC * 1000.0
    rt_ok = float(np.mean(np.asarray(e2e_ms) <= budget_ms)) if e2e_ms else float("nan")
    rest_p50 = (e2e["p50"] - embed["p50"]
                if np.isfinite(e2e["p50"]) and np.isfinite(embed["p50"]) else float("nan"))
    return {
        "budget_ms": budget_ms,
        "rt_ok_pct": rt_ok,
        "headroom_p95_ms": budget_ms - e2e["p95"] if np.isfinite(e2e["p95"]) else float("nan"),
        "rest_p50_ms": rest_p50,
        "end_to_end": e2e,
        "embed": embed,
    }


def print_report(aligner: SongAligner, r: dict) -> None:
    artist = f"{aligner.artist} — " if aligner.artist else ""
    print(f"\nLatency: {artist}{aligner.song_id}  [{aligner.song_slug}]  "
          f"matcher={aligner.matcher}")
    print(f"  budget {r['budget_ms']:.0f}ms/chunk   "
          f"rt-ok {r['rt_ok_pct'] * 100:.1f}%   "
          f"p95 headroom {r['headroom_p95_ms']:+.0f}ms")
    hdr = f"  {'stage':<12} {'mean':>7} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}   n"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for label, key in (("embed", "embed"), ("end-to-end", "end_to_end")):
        s = r[key]
        print(f"  {label:<12} {s['mean']:7.1f} {s['p50']:7.1f} {s['p90']:7.1f} "
              f"{s['p95']:7.1f} {s['p99']:7.1f} {s['max']:7.1f}   {s['n']}")
    print(f"  {'rest (p50)':<12} {r['rest_p50_ms']:7.1f}   "
          f"(matcher + HMM + trigger = end-to-end - embed)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-chunk latency benchmark.")
    p.add_argument("cache", help="Path to .npz cache from ppsync-preprocess.")
    p.add_argument("--file", required=True, help="Audio file to replay.")
    p.add_argument("--matcher", default=None, choices=("dtw", "rigid"),
                   help="Sequence matcher (default: config.MATCHER).")
    p.add_argument("--dtw-step-penalty", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--warmup-chunks", type=int, default=60,
                   help="Scored chunks to discard before measuring (warm-up).")
    p.add_argument("--measure-chunks", type=int, default=300,
                   help="Scored chunks to measure after warm-up.")
    p.add_argument("--embed-iters", type=int, default=200,
                   help="Isolated MERT-forward timing iterations.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    device = _pick_device(args.device)
    print(f"Loading MERT on {device}…")
    processor, model = load_model(device)
    step_penalty = (args.dtw_step_penalty if args.dtw_step_penalty is not None
                    else DTW_STEP_PENALTY)
    aligner = SongAligner(
        cache_path=Path(args.cache), model=model, processor=processor,
        device=device, dry_run=True, wall_timers=False,
        matcher=args.matcher or MATCHER, dtw_step_penalty=step_penalty,
    )

    r = measure(aligner, Path(args.file), args.warmup_chunks,
                args.measure_chunks, args.embed_iters)
    print_report(aligner, r)

    if args.json_out:
        payload = {
            "artist": aligner.artist,
            "song_id": aligner.song_id,
            "song_slug": aligner.song_slug,
            "audio_file": str(args.file),
            "matcher": aligner.matcher,
            "device": device,
            "date": datetime.now().isoformat(timespec="seconds"),
            "latency": r,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
