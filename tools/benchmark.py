"""Offline benchmark: how well does the aligner re-sync when the song starts
at an arbitrary point?

This sweeps a set of start offsets over a single test audio file, replaying each
through a fresh SongAligner (model loaded once, reused across offsets), and
reports both *tracking* accuracy (does the aligner know where it is in the
song?) and *trigger* accuracy (do slides fire at the right moment?).

Why studio-against-itself:  the live/spoken annotation JSONs reuse the studio
timestamps verbatim, so only the studio audio has ground-truth timing that
matches its own performance.  Replaying studio audio at offset T simulates
"the operator joined the song T seconds in" while keeping exact ground truth —
the true song position of every chunk is simply its file position.

Tracking metrics per offset (per-frame, after warmup):
    dtw_med / dtw_p90    median / 90th-pct of |dtw_refined_t - true_t|  (s)
    hmm_med / hmm_p90    same for |hmm_expected_pos_t - true_t|         (s)
    track%               fraction of frames with DTW error <= 1.0s
    lock_s               seconds after start until DTW error first <= 1.0s

Trigger metrics per offset (only *reachable* boundaries are scored — a
boundary before the start offset or inside the warmup window can never be
triggered and is excluded rather than counted as a miss):
    reachable      boundaries with t_ref >= offset + warmup
    hits           reachable boundaries whose trigger FIRED within --window-ms
                   of the true boundary (fire time, not predicted time)
    recall         hits / reachable
    fire MAE/max   |fire_t - true boundary| over hits  (ms)
    spurious       triggers that match no reachable boundary in-window

Usage:
    python tools/benchmark.py data/studio_cache.npz \
        --file /Users/das/propresenter-dataset/incubus/drive/studio_drive.wav \
        --manifest data/studio_manifest.json \
        --offsets 0,30,64,95,130,170

    # Partial-song test: play only 40s starting at 64s
    python tools/benchmark.py ... --offsets 64 --duration 40

    # Per-frame telemetry dump for diagnosis (single offset recommended)
    python tools/benchmark.py ... --offsets 30 --trace-out /tmp/trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from ppsync.aligner import SongAligner
from ppsync.audio_capture import FileCapture
from ppsync.config import CHUNK_SEC, DTW_LIVE_SEC, LOOKBACK_SEC
from ppsync.embed import load_model


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_offset(
    aligner: SongAligner,
    audio_path: Path,
    offset: float,
    gt: list[tuple[str, float]],
    warmup_sec: float,
    window_ms: float,
    duration: float | None = None,
    trace_path: Path | None = None,
    total_chunks: int | None = None,
) -> dict:
    """Replay the file from *offset* and score the aligner against true file time."""
    aligner.reset()
    source = FileCapture(audio_path=audio_path, realtime=False, start_offset_sec=offset)

    triggers: list[dict] = []   # {slide_id, fire_t}
    trace: list[dict] = []
    dtw_errs: list[float] = []  # |dtw_refined_t - true song time|, post-warmup
    hmm_errs: list[float] = []
    lock_on_t: float | None = None
    proc_ms: list[float] = []   # per-chunk processing latency

    end_t = None if duration is None else offset + duration
    horizon = offset + warmup_sec

    bar = tqdm(source, total=total_chunks, unit="chunk", leave=False,
               desc=f"offset {offset:g}s", dynamic_ncols=True)
    for chunk, song_t in bar:
        if end_t is not None and song_t >= end_t:
            bar.close()
            break
        frame = aligner.process_chunk(chunk, chunk_wall_t=song_t)
        if frame.get("status"):  # buffering / silence
            continue
        proc_ms.append(frame.get("processing_ms", 0.0))

        dtw_err = frame["dtw_refined_t"] - song_t
        hmm_err = frame["hmm_expected_pos_t"] - song_t
        if song_t >= horizon:
            dtw_errs.append(abs(dtw_err))
            hmm_errs.append(abs(hmm_err))
        if lock_on_t is None and abs(dtw_err) <= 1.0:
            lock_on_t = song_t
        if frame["triggered"]:
            triggers.append({"slide_id": frame["triggered_slide_id"], "fire_t": song_t})
        if trace_path is not None:
            frame["true_t"] = round(song_t, 3)
            trace.append(frame)

    # Reachable ground-truth boundaries (exclude pre-offset + warmup region).
    reachable = [(sid, t) for sid, t in gt if t >= horizon]
    if end_t is not None:
        reachable = [(sid, t) for sid, t in reachable if t <= end_t]
    gt_by_id = dict(gt)
    window_sec = window_ms / 1000.0

    matched_ids: set[str] = set()
    fire_errors_ms: list[float] = []
    spurious = 0
    for trig in triggers:
        sid = trig["slide_id"]
        if sid in gt_by_id and gt_by_id[sid] < horizon:
            continue  # catch-up fire for a pre-warmup boundary — by design
        err = abs(trig["fire_t"] - gt_by_id[sid]) if sid in gt_by_id else float("inf")
        if err <= window_sec and sid not in matched_ids:
            matched_ids.add(sid)
            fire_errors_ms.append(err * 1000.0)
        else:
            spurious += 1

    reachable_ids = {sid for sid, _ in reachable}
    hits = len(matched_ids & reachable_ids)

    if trace_path is not None:
        trace_path.write_text(json.dumps(trace, indent=1))
        print(f"  wrote {len(trace)} frames to {trace_path}")

    dtw_a = np.array(dtw_errs) if dtw_errs else np.array([np.nan])
    hmm_a = np.array(hmm_errs) if hmm_errs else np.array([np.nan])
    return {
        "offset": offset,
        "duration": duration,
        # tracking
        "dtw_med_s": float(np.median(dtw_a)),
        "dtw_p90_s": float(np.percentile(dtw_a, 90)),
        "hmm_med_s": float(np.median(hmm_a)),
        "hmm_p90_s": float(np.percentile(hmm_a, 90)),
        "track_pct": float(np.mean(dtw_a <= 1.0)),
        "lock_on_s": (lock_on_t - offset) if lock_on_t is not None else float("nan"),
        # triggers
        "reachable": len(reachable),
        "hits": hits,
        "recall": hits / len(reachable) if reachable else float("nan"),
        "fire_mae_ms": float(np.mean(fire_errors_ms)) if fire_errors_ms else float("nan"),
        "fire_max_ms": float(np.max(fire_errors_ms)) if fire_errors_ms else float("nan"),
        "spurious": spurious,
        "n_triggers": len(triggers),
        "proc_mean_ms": float(np.mean(proc_ms)) if proc_ms else float("nan"),
        "proc_p50_ms": float(np.percentile(proc_ms, 50)) if proc_ms else float("nan"),
        "proc_p95_ms": float(np.percentile(proc_ms, 95)) if proc_ms else float("nan"),
        "max_proc_ms": float(np.max(proc_ms)) if proc_ms else float("nan"),
        # fraction of chunks processed faster than real time (chunk duration)
        "rt_ok_pct": float(np.mean(np.array(proc_ms) <= CHUNK_SEC * 1000.0))
                     if proc_ms else float("nan"),
        "missed": sorted(reachable_ids - matched_ids),
        "triggered": [(t["slide_id"], round(t["fire_t"], 2)) for t in triggers],
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Start-offset re-sync benchmark.")
    p.add_argument("cache", help="Path to .npz cache from ppsync-preprocess.")
    p.add_argument("--file", required=True, help="Test audio file.")
    p.add_argument("--manifest", required=True, help="Ground-truth manifest JSON.")
    p.add_argument("--offsets", default="0,30,64,95,130,170",
                   help="Comma-separated start offsets in seconds.")
    p.add_argument("--duration", type=float, default=None,
                   help="Play only this many seconds from each offset (partial-song test).")
    p.add_argument("--window-ms", type=float, default=750.0,
                   help="Trigger-correct tolerance window on fire time (default 750ms).")
    p.add_argument("--warmup-sec", type=float, default=LOOKBACK_SEC + DTW_LIVE_SEC,
                   help="Cold-start window after offset; boundaries inside it are unscored.")
    p.add_argument("--device", default=None)
    p.add_argument("--json-out", default=None, help="Write raw per-offset results to JSON.")
    p.add_argument("--trace-out", default=None,
                   help="Write per-frame telemetry JSON (suffixed per offset).")
    args = p.parse_args(argv)

    offsets = [float(x) for x in args.offsets.split(",") if x.strip() != ""]
    gt = [(s["slide_id"], float(s["t_ref"]))
          for s in json.loads(Path(args.manifest).read_text())["slides"]]

    audio_dur = sf.info(args.file).duration
    chunks_per_offset = [
        int(max(0.0, (min(audio_dur, o + args.duration) if args.duration else audio_dur) - o)
            / CHUNK_SEC)
        for o in offsets
    ]
    # ~15 chunks/s is typical on Apple Silicon MPS once warm; first chunk pays
    # model warm-up (~3s).  This is only a rough upfront estimate.
    est_s = sum(chunks_per_offset) / 15.0 + 5.0
    print(f"Replaying {sum(chunks_per_offset)} chunks across {len(offsets)} offsets "
          f"— rough estimate {est_s/60.0:.1f} min (progress bar per offset below).")

    device = _pick_device(args.device)
    print(f"Loading MERT on {device}…")
    processor, model = load_model(device)
    aligner = SongAligner(
        cache_path=Path(args.cache),
        model=model, processor=processor, device=device, dry_run=True,
    )

    print(f"\nBenchmark: {Path(args.file).name}  "
          f"({len(gt)} boundaries, window={args.window_ms:.0f}ms, "
          f"warmup={args.warmup_sec:.1f}s"
          + (f", duration={args.duration:.0f}s" if args.duration else "") + ")\n")
    header = (f"{'offset':>7} {'lock_s':>7} {'dtw_med':>8} {'dtw_p90':>8} {'track%':>7} "
              f"{'reach':>6} {'hits':>5} {'recall':>7} {'fireMAE':>8} {'spur':>5}")
    print(header)
    print("-" * len(header))

    results = []
    for off, n_chunks in zip(offsets, chunks_per_offset):
        trace_path = None
        if args.trace_out:
            tp = Path(args.trace_out)
            trace_path = tp.with_name(f"{tp.stem}_off{off:g}{tp.suffix}")
        r = run_offset(aligner, Path(args.file), off, gt, args.warmup_sec,
                       args.window_ms, duration=args.duration,
                       trace_path=trace_path, total_chunks=n_chunks)
        results.append(r)
        print(f"{r['offset']:7.0f} {r['lock_on_s']:7.2f} {r['dtw_med_s']:8.2f} "
              f"{r['dtw_p90_s']:8.2f} {r['track_pct']:7.2f} "
              f"{r['reachable']:6d} {r['hits']:5d} {r['recall']:7.2f} "
              f"{r['fire_mae_ms']:8.1f} {r['spurious']:5d}")
        if r["triggered"]:
            print(f"        fired: {', '.join(f'{s}@{t}' for s, t in r['triggered'])}")
        if r["missed"]:
            print(f"        missed: {', '.join(r['missed'])}")

    recalls = [r["recall"] for r in results if not np.isnan(r["recall"])]
    print("-" * len(header))
    print(f"mean recall {np.mean(recalls):.2f}   "
          f"mean dtw_med {np.mean([r['dtw_med_s'] for r in results]):.2f}s")
    print(f"latency/chunk (budget {CHUNK_SEC * 1000:.0f}ms): "
          f"mean {np.mean([r['proc_mean_ms'] for r in results]):.1f}  "
          f"p50 {np.mean([r['proc_p50_ms'] for r in results]):.1f}  "
          f"p95 {np.mean([r['proc_p95_ms'] for r in results]):.1f}  "
          f"max {max(r['max_proc_ms'] for r in results):.1f}  "
          f"rt-ok {100 * np.mean([r['rt_ok_pct'] for r in results]):.1f}%")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
