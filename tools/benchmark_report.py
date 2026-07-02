"""Run the accuracy + latency benchmarks across a whole dataset and emit a
single Markdown report (plus a machine-readable JSON) for check-in / CI.

For every song under --dataset that has BOTH an audio file and a ProPresenter
annotation JSON, this:

  1. builds a ppsync manifest (tools/pp_to_manifest.convert) pointing at the
     dataset audio,
  2. ensures an embedding cache exists (ppsync-preprocess; reused if present
     unless --rebuild) — the slow step, one MERT sliding-window pass per song,
  3. scores ACCURACY by replaying the song from each --offset through the real
     SongAligner (tools/benchmark.run_offset),
  4. measures per-chunk LATENCY (tools/latency_benchmark.measure),

then aggregates per-song and overall into benchmarks/REPORT.md + results.json.

The MERT model is loaded once and reused for every song.  Caches/manifests
land in the gitignored data/ tree (rebuildable); only the report + results
JSON are meant to be committed.

Why studio-against-itself: each song's annotation timestamps describe its own
audio, so replaying that audio gives exact per-chunk ground truth (the true
song position of a chunk is its file position).  See tools/benchmark.py.

Usage:
    python tools/benchmark_report.py --dataset ../slide-agent/dataset
    python tools/benchmark_report.py --dataset <dir> --offsets 0,30 --matcher rigid
    python tools/benchmark_report.py --dataset <dir> --only cocaine,layla --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# tools/ is not a package — make sibling modules importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as accuracy  # noqa: E402
import latency_benchmark as latency  # noqa: E402
import pp_to_manifest  # noqa: E402

from ppsync.aligner import SongAligner  # noqa: E402
from ppsync.config import (  # noqa: E402
    CHUNK_SEC,
    DTW_LIVE_SEC,
    DTW_STEP_PENALTY,
    LOOKBACK_SEC,
    MATCHER,
)
from ppsync.embed import load_model  # noqa: E402
from ppsync.io import song_dir, song_slug  # noqa: E402
from ppsync.preprocess import preprocess_song  # noqa: E402


def discover_songs(dataset: Path) -> list[dict]:
    """Find <dataset>/<artist>/<song>/ dirs containing both audio and a JSON.

    Returns dicts with artist (prettified from the folder), audio path, and
    annotation-JSON path, sorted by artist then song.
    """
    audio_exts = {".wav", ".flac", ".mp3"}
    songs: list[dict] = []
    for artist_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        for song_subdir in sorted(p for p in artist_dir.iterdir() if p.is_dir()):
            audio = next((f for f in sorted(song_subdir.iterdir())
                          if f.suffix.lower() in audio_exts), None)
            jsons = [f for f in sorted(song_subdir.iterdir())
                     if f.suffix.lower() == ".json"]
            if audio is None or not jsons:
                continue  # no wav -> ignore (per user); no annotation -> skip
            songs.append({
                "artist": artist_dir.name.replace("-", " ").title(),
                "audio": audio,
                "annotation": jsons[0],
                "folder": song_subdir.name,
            })
    return songs


def ensure_cache(song: dict, data_dir: str, rebuild: bool) -> tuple[Path, Path]:
    """Build the manifest (always) and embedding cache (if missing/rebuild)."""
    manifest = pp_to_manifest.convert(
        song["annotation"], artist=song["artist"],
        audio_override=str(song["audio"].resolve()),
    )
    slug = song_slug(manifest["artist"], manifest["song_id"])
    out_dir = song_dir(manifest["artist"], manifest["song_id"], data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{slug}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    cache_path = out_dir / f"{slug}_cache.npz"
    if rebuild or not cache_path.exists():
        preprocess_song(manifest_path=manifest_path, output_path=cache_path,
                        show_progress=sys.stderr.isatty())
    return manifest_path, cache_path


def run_song(song: dict, model, processor, device: str, data_dir: str,
             offsets: list[float], matcher: str, step_penalty: float,
             window_ms: float, warmup_sec: float, rebuild: bool,
             lat_warmup: int, lat_measure: int, lat_iters: int) -> dict:
    """Accuracy (per offset) + latency for one song; aligner reused across both."""
    manifest_path, cache_path = ensure_cache(song, data_dir, rebuild)
    gt = [(s["slide_id"], float(s["t_ref"]))
          for s in json.loads(manifest_path.read_text())["slides"]]

    aligner = SongAligner(
        cache_path=cache_path, model=model, processor=processor, device=device,
        dry_run=True, wall_timers=False, matcher=matcher,
        dtw_step_penalty=step_penalty,
    )

    offset_results = [
        accuracy.run_offset(aligner, song["audio"], off, gt, warmup_sec,
                            window_ms, duration=None)
        for off in offsets
    ]
    lat = latency.measure(aligner, song["audio"], lat_warmup, lat_measure, lat_iters)

    # Aggregate accuracy across offsets: recall is hits/reachable pooled;
    # fire error / tracking are averaged over offsets that produced a value.
    hits = sum(r["hits"] for r in offset_results)
    reachable = sum(r["reachable"] for r in offset_results)
    spurious = sum(r["spurious"] for r in offset_results)
    fire_maes = [r["fire_mae_ms"] for r in offset_results if np.isfinite(r["fire_mae_ms"])]
    fire_maxes = [r["fire_max_ms"] for r in offset_results if np.isfinite(r["fire_max_ms"])]
    locks = [r["lock_on_s"] for r in offset_results if np.isfinite(r["lock_on_s"])]
    return {
        "artist": aligner.artist,
        "song_id": aligner.song_id,
        "slug": aligner.song_slug,
        "n_boundaries": len(gt),
        "song_duration": aligner.song_duration,
        "offsets": offsets,
        # accuracy (pooled over offsets)
        "recall": hits / reachable if reachable else float("nan"),
        "hits": hits,
        "reachable": reachable,
        "spurious": spurious,
        "fire_mae_ms": float(np.mean(fire_maes)) if fire_maes else float("nan"),
        "fire_max_ms": float(np.max(fire_maxes)) if fire_maxes else float("nan"),
        "track_med_s": float(np.mean([r["dtw_med_s"] for r in offset_results])),
        "lock_on_s": float(np.mean(locks)) if locks else float("nan"),
        # latency
        "lat_embed_p50_ms": lat["embed"]["p50"],
        "lat_e2e_p50_ms": lat["end_to_end"]["p50"],
        "lat_e2e_p95_ms": lat["end_to_end"]["p95"],
        "lat_e2e_max_ms": lat["end_to_end"]["max"],
        "lat_rt_ok_pct": lat["rt_ok_pct"],
        "per_offset": offset_results,
    }


def _f(x: float, fmt: str, na: str = "—") -> str:
    return na if x is None or not np.isfinite(x) else format(x, fmt)


def render_markdown(meta: dict, rows: list[dict]) -> str:
    L = []
    L.append("# ppsync benchmark report")
    L.append("")
    L.append("_Generated by `tools/benchmark_report.py` — do not edit by hand._")
    L.append("")
    L.append(f"- **Date:** {meta['date']}")
    L.append(f"- **Device:** `{meta['device']}`  ·  **Matcher:** `{meta['matcher']}`"
             f"  ·  **DTW step penalty:** {meta['step_penalty']}")
    L.append(f"- **Songs:** {len(rows)}  ·  **Start offsets:** "
             f"{', '.join(f'{o:g}s' for o in meta['offsets'])}")
    L.append(f"- **Trigger window:** ±{meta['window_ms']:.0f}ms  ·  "
             f"**Warm-up:** {meta['warmup_sec']:.1f}s  ·  "
             f"**Chunk budget:** {CHUNK_SEC * 1000:.0f}ms")
    L.append("")

    # ---- overall ----
    recalls = [r["recall"] for r in rows if np.isfinite(r["recall"])]
    tot_hits = sum(r["hits"] for r in rows)
    tot_reach = sum(r["reachable"] for r in rows)
    tot_spur = sum(r["spurious"] for r in rows)
    maes = [r["fire_mae_ms"] for r in rows if np.isfinite(r["fire_mae_ms"])]
    e2e_p95 = [r["lat_e2e_p95_ms"] for r in rows if np.isfinite(r["lat_e2e_p95_ms"])]
    rt_oks = [r["lat_rt_ok_pct"] for r in rows if np.isfinite(r["lat_rt_ok_pct"])]
    L.append("## Summary")
    L.append("")
    L.append(f"- **Trigger recall:** {tot_hits}/{tot_reach} reachable boundaries "
             f"on time (**{100 * tot_hits / tot_reach:.1f}%**)"
             if tot_reach else "- **Trigger recall:** n/a")
    L.append(f"- **Mean fire error:** {_f(float(np.mean(maes)), '.0f')}ms  ·  "
             f"**Spurious fires:** {tot_spur}")
    L.append(f"- **Per-song recall:** min {_f(min(recalls) * 100, '.0f')}% · "
             f"median {_f(float(np.median(recalls)) * 100, '.0f')}% · "
             f"max {_f(max(recalls) * 100, '.0f')}%" if recalls else "")
    L.append(f"- **Latency:** worst-song p95 {_f(max(e2e_p95), '.1f')}ms vs "
             f"{CHUNK_SEC * 1000:.0f}ms budget  ·  "
             f"real-time-OK {_f(100 * float(np.mean(rt_oks)), '.1f')}% of chunks")
    L.append("")

    # ---- accuracy table ----
    unscorable = [r for r in rows if r["reachable"] == 0]
    L.append("## Accuracy")
    L.append("")
    L.append("| Song | Artist | Slides | Recall | Fire MAE | Fire max | Track med | Lock-on | Spurious |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda x: x["slug"]):
        if r["reachable"] == 0:  # source annotation has no usable slide timings
            L.append(
                f"| {r['song_id']} | {r['artist']} | {r['n_boundaries']} "
                f"| n/a † | n/a | n/a "
                f"| {_f(r['track_med_s'], '.2f')}s "
                f"| {_f(r['lock_on_s'], '.1f')}s | {r['spurious']} |"
            )
            continue
        L.append(
            f"| {r['song_id']} | {r['artist']} | {r['n_boundaries']} "
            f"| {_f(r['recall'] * 100, '.0f')}% "
            f"| {_f(r['fire_mae_ms'], '.0f')}ms "
            f"| {_f(r['fire_max_ms'], '.0f')}ms "
            f"| {_f(r['track_med_s'], '.2f')}s "
            f"| {_f(r['lock_on_s'], '.1f')}s "
            f"| {r['spurious']} |"
        )
    L.append("")
    L.append("_Recall = reachable boundaries fired within the trigger window, "
             "pooled over offsets. Fire MAE/max = |fire − true boundary|. "
             "Track med = median |position − truth|. Lock-on = seconds from "
             "start to first within-1s tracking._")
    if unscorable:
        names = ", ".join(sorted(r["song_id"] for r in unscorable))
        L.append("")
        L.append(f"_† **Unscorable accuracy** ({names}): the source annotation "
                 f"carries no slide timings (all trigger times are 0), so there "
                 f"is no ground truth to score against. Excluded from the recall "
                 f"totals above; latency below is still valid._")
    L.append("")

    # ---- latency table ----
    L.append("## Latency (per chunk)")
    L.append("")
    L.append("| Song | Embed p50 | End-to-end p50 | p95 | max | RT-OK |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda x: x["slug"]):
        L.append(
            f"| {r['song_id']} "
            f"| {_f(r['lat_embed_p50_ms'], '.1f')}ms "
            f"| {_f(r['lat_e2e_p50_ms'], '.1f')}ms "
            f"| {_f(r['lat_e2e_p95_ms'], '.1f')}ms "
            f"| {_f(r['lat_e2e_max_ms'], '.1f')}ms "
            f"| {_f(r['lat_rt_ok_pct'] * 100, '.1f')}% |"
        )
    L.append("")
    L.append(f"_End-to-end = full `process_chunk()`. Embed = isolated MERT "
             f"forward (the dominant stage). RT-OK = chunks processed within "
             f"the {CHUNK_SEC * 1000:.0f}ms budget._")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Dataset-wide accuracy + latency report.")
    p.add_argument("--dataset", required=True,
                   help="Dataset root: <artist>/<song>/ with audio + annotation JSON.")
    p.add_argument("--offsets", default="0",
                   help="Comma-separated start offsets in seconds (default: 0).")
    p.add_argument("--matcher", default=None, choices=("dtw", "rigid"),
                   help="Sequence matcher (default: config.MATCHER).")
    p.add_argument("--dtw-step-penalty", type=float, default=None)
    p.add_argument("--window-ms", type=float, default=750.0)
    p.add_argument("--warmup-sec", type=float, default=LOOKBACK_SEC + DTW_LIVE_SEC)
    p.add_argument("--only", default=None,
                   help="Comma-separated song-folder substrings to include.")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild embedding caches even if present.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--lat-warmup", type=int, default=60)
    p.add_argument("--lat-measure", type=int, default=300)
    p.add_argument("--lat-iters", type=int, default=200)
    p.add_argument("--out-md", default="benchmarks/REPORT.md")
    p.add_argument("--out-json", default="benchmarks/results.json")
    p.add_argument("--from-json", default=None,
                   help="Skip running; re-render REPORT.md from an existing "
                        "results.json (e.g. after tweaking the renderer).")
    args = p.parse_args(argv)

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text())
        md = render_markdown(data["meta"], data["songs"])
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md)
        print(f"Re-rendered {out_md} from {args.from_json} "
              f"({len(data['songs'])} songs)")
        return

    offsets = [float(x) for x in args.offsets.split(",") if x.strip() != ""]
    dataset = Path(args.dataset)
    songs = discover_songs(dataset)
    if args.only:
        wanted = [s.strip().lower() for s in args.only.split(",")]
        songs = [s for s in songs
                 if any(w in s["folder"].lower() for w in wanted)]
    if not songs:
        raise SystemExit(f"No songs with audio + annotation found under {dataset}")

    device = latency._pick_device(args.device)
    matcher = args.matcher or MATCHER
    step_penalty = (args.dtw_step_penalty if args.dtw_step_penalty is not None
                    else DTW_STEP_PENALTY)
    print(f"Loading MERT on {device}…  ({len(songs)} songs, matcher={matcher})")
    processor, model = load_model(device)

    rows: list[dict] = []
    for i, song in enumerate(songs, 1):
        dur = sf.info(str(song["audio"])).duration
        print(f"[{i}/{len(songs)}] {song['artist']} — {song['folder']} "
              f"({dur:.0f}s)…", flush=True)
        try:
            rows.append(run_song(
                song, model, processor, device, args.data_dir, offsets,
                matcher, step_penalty, args.window_ms, args.warmup_sec,
                args.rebuild, args.lat_warmup, args.lat_measure, args.lat_iters,
            ))
        except Exception as exc:  # one bad song must not sink the whole report
            print(f"    SKIPPED ({type(exc).__name__}: {exc})", flush=True)

    if not rows:
        raise SystemExit("No songs produced results.")

    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "matcher": matcher,
        "step_penalty": step_penalty,
        "offsets": offsets,
        "window_ms": args.window_ms,
        "warmup_sec": args.warmup_sec,
        "chunk_budget_ms": CHUNK_SEC * 1000.0,
    }
    md = render_markdown(meta, rows)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"meta": meta, "songs": rows}, indent=2))
    print(f"\nWrote {out_md}  and  {args.out_json}  ({len(rows)} songs)")


if __name__ == "__main__":
    main()
