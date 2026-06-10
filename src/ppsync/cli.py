"""CLI entry points for ppsync.

Commands
--------
ppsync-preprocess   Offline preprocessing: build reference embedding cache.
ppsync-align        Live alignment (mic or file).
ppsync-eval         Offline evaluation against ground-truth annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    CHUNK_SEC,
    DTW_LIVE_SEC,
    DTW_SEARCH_SEC,
    LOOKBACK_SEC,
    MERT_LAYER,
    REST_URL,
    STRIDE_SEC,
    TARGET_SR,
    TRIGGER_BUFFER_MS,
    TRIGGER_CONFIDENCE_MIN,
)


# ===========================================================================
# ppsync-preprocess
# ===========================================================================

def _preprocess_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ppsync-preprocess",
        description=(
            "Offline preprocessing: run MERT over the reference audio and "
            "save a dense embedding cache (.npz) for use by ppsync-align."
        ),
    )
    p.add_argument("manifest", help="Path to slide manifest JSON.")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output .npz path (default: <manifest_stem>.npz beside the JSON).",
    )
    p.add_argument(
        "--lookback", type=float, default=LOOKBACK_SEC, metavar="SEC",
        help=f"Sliding window lookback in seconds (default: {LOOKBACK_SEC}).",
    )
    p.add_argument(
        "--stride", type=float, default=STRIDE_SEC, metavar="SEC",
        help=f"Window stride in seconds (default: {STRIDE_SEC}).  "
             "Smaller = denser reference, slower preprocessing.",
    )
    p.add_argument(
        "--layer", type=int, default=MERT_LAYER, metavar="N",
        help=f"MERT transformer layer to use (0=CNN, 1-12=transformer; default: {MERT_LAYER}).",
    )
    p.add_argument(
        "--device", default=None,
        help="Compute device: cpu | cuda | mps (auto-detected by default).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return p


def preprocess_main(argv: list[str] | None = None) -> None:
    from .preprocess import preprocess_song

    args = _preprocess_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    output_path = (
        Path(args.output)
        if args.output
        else manifest_path.with_suffix(".npz")
    )
    preprocess_song(
        manifest_path=manifest_path,
        output_path=output_path,
        lookback_sec=args.lookback,
        stride_sec=args.stride,
        mert_layer=args.layer,
        device=args.device,
        show_progress=not args.quiet,
    )


# ===========================================================================
# ppsync-align
# ===========================================================================

def _align_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ppsync-align",
        description=(
            "Run real-time music alignment.  Audio source is either the "
            "system microphone or a WAV/FLAC file (for testing)."
        ),
    )
    p.add_argument("cache", help="Path to .npz cache from ppsync-preprocess.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="Capture from system microphone.")
    src.add_argument("--file", metavar="AUDIO", help="Simulate real-time from a file.")

    p.add_argument(
        "--start-offset", type=float, default=0.0, metavar="SEC",
        help="Skip this many seconds of the file before starting (tests cold-start sync).",
    )
    p.add_argument(
        "--no-realtime", action="store_true",
        help="With --file: process as fast as possible (ignores real-time pacing).",
    )
    p.add_argument(
        "--rest-url", default=REST_URL,
        help=f"REST endpoint for slide triggers (default: {REST_URL}).",
    )
    p.add_argument(
        "--trigger-buffer", type=float, default=TRIGGER_BUFFER_MS, metavar="MS",
        help=f"Fire trigger this many ms before slide boundary (default: {TRIGGER_BUFFER_MS}).",
    )
    p.add_argument(
        "--trigger-conf", type=float, default=TRIGGER_CONFIDENCE_MIN, metavar="FLOAT",
        help=f"Minimum HMM trigger confidence to fire (default: {TRIGGER_CONFIDENCE_MIN}).",
    )
    p.add_argument(
        "--chunk", type=float, default=CHUNK_SEC, metavar="SEC",
        help=f"Audio chunk size in seconds (default: {CHUNK_SEC}).",
    )
    p.add_argument(
        "--dtw-live", type=float, default=DTW_LIVE_SEC, metavar="SEC",
        help=f"Live buffer fed to DTW in seconds (default: {DTW_LIVE_SEC}).",
    )
    p.add_argument(
        "--dtw-search", type=float, default=DTW_SEARCH_SEC, metavar="SEC",
        help=f"Forward search window in reference in seconds (default: {DTW_SEARCH_SEC}).",
    )
    p.add_argument(
        "--log", default=None, metavar="FILE",
        help="Write JSON-lines telemetry to FILE.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print trigger events but do not send HTTP requests.",
    )
    p.add_argument(
        "--device", default=None,
        help="Compute device: cpu | cuda | mps.",
    )
    return p


def align_main(argv: list[str] | None = None) -> None:
    import torch

    from .aligner import SongAligner
    from .audio_capture import FileCapture, MicCapture
    from .embed import load_model
    from .telemetry import TelemetryLogger

    args = _align_parser().parse_args(argv)

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading MERT on {device}…")
    processor, model = load_model(device)

    cache_path = Path(args.cache)
    aligner = SongAligner(
        cache_path=cache_path,
        model=model,
        processor=processor,
        device=device,
        rest_url=args.rest_url,
        trigger_buffer_ms=args.trigger_buffer,
        trigger_conf_min=args.trigger_conf,
        dry_run=args.dry_run,
        dtw_live_sec=args.dtw_live,
        dtw_search_sec=args.dtw_search,
        chunk_sec=args.chunk,
    )

    if args.mic:
        source = MicCapture(chunk_sec=args.chunk)
        print("Listening on microphone — Ctrl+C to stop.\n")
    else:
        source = FileCapture(
            audio_path=Path(args.file),
            chunk_sec=args.chunk,
            realtime=not args.no_realtime,
            start_offset_sec=args.start_offset,
        )
        print(f"Processing file: {args.file}  (offset: {args.start_offset:.1f}s)\n")

    log_path = Path(args.log) if args.log else None

    with TelemetryLogger(log_path) as logger:
        try:
            for chunk, wall_t in source:
                frame = aligner.process_chunk(chunk, chunk_wall_t=wall_t)
                logger.log(frame)
                if frame.get("status") == "buffering":
                    continue
                status = (
                    f"  chunk={frame['chunk']:4d}"
                    f"  dtw_t={frame['dtw_refined_t']:6.2f}s"
                    f"  dtw_conf={frame['dtw_confidence']:.2f}"
                    f"  slide=[{frame['hmm_current_slide_id']}]"
                    f"  trigger_conf={frame['hmm_trigger_confidence']:.2f}"
                )
                if frame["triggered"]:
                    status += f"  *** TRIGGER → {frame['triggered_slide_id']} ***"
                print(status)
        except KeyboardInterrupt:
            print("\nStopped.")


# ===========================================================================
# ppsync-eval
# ===========================================================================

def _eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ppsync-eval",
        description=(
            "Offline evaluation: replay a test audio file through the "
            "alignment pipeline and compare predicted trigger times against "
            "a ground-truth annotation JSON."
        ),
    )
    p.add_argument("cache", help="Path to .npz cache from ppsync-preprocess.")
    p.add_argument("--file", required=True, metavar="AUDIO",
                   help="Test audio file (alternate performance, same song).")
    p.add_argument(
        "--ground-truth", required=True, metavar="JSON",
        help="Ground-truth slide annotation JSON in the same format as the manifest.",
    )
    p.add_argument(
        "--start-offset", type=float, default=0.0, metavar="SEC",
        help="Skip this many seconds of the test file.",
    )
    p.add_argument(
        "--log", default=None, metavar="FILE",
        help="Write JSON-lines telemetry to FILE.",
    )
    p.add_argument(
        "--window-ms", type=float, default=500.0, metavar="MS",
        help="Tolerance window for counting triggers as correct (default: 500ms).",
    )
    p.add_argument("--device", default=None)
    return p


def eval_main(argv: list[str] | None = None) -> None:
    import torch

    from .aligner import SongAligner
    from .audio_capture import FileCapture
    from .embed import load_model
    from .io import finalize_slide_stops, load_audio, load_manifest
    from .telemetry import TelemetryLogger

    args = _eval_parser().parse_args(argv)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load ground-truth annotations
    gt_audio, gt_slides = load_manifest(Path(args.ground_truth))
    gt_wav = load_audio(gt_audio if gt_audio.exists() else Path(args.file))
    finalize_slide_stops(gt_slides, gt_wav.shape[0] / TARGET_SR)
    gt_t_refs = {s["slide_id"]: s["t_ref"] for s in gt_slides}

    print(f"Ground truth: {len(gt_slides)} slides")

    print(f"Loading MERT on {device}…")
    processor, model = load_model(device)

    aligner = SongAligner(
        cache_path=Path(args.cache),
        model=model,
        processor=processor,
        device=device,
        dry_run=True,  # eval never fires real HTTP
    )

    source = FileCapture(
        audio_path=Path(args.file),
        realtime=False,
        start_offset_sec=args.start_offset,
    )

    log_path = Path(args.log) if args.log else None
    trigger_log: list[dict] = []

    with TelemetryLogger(log_path) as logger:
        for chunk, wall_t in source:
            frame = aligner.process_chunk(chunk, chunk_wall_t=wall_t)
            logger.log(frame)
            if frame.get("triggered"):
                trigger_log.append(
                    {
                        "slide_id": frame["triggered_slide_id"],
                        "pred_t": frame["dtw_refined_t"],
                        "boundary_t": frame["hmm_predicted_next_t"],
                    }
                )

    # --- Evaluation metrics ---
    window_sec = args.window_ms / 1000.0
    tp, fp, fn = 0, 0, 0
    errors_ms: list[float] = []

    matched_gt = set()
    for trig in trigger_log:
        sid = trig["slide_id"]
        pred_t = trig["boundary_t"]
        if sid in gt_t_refs:
            gt_t = gt_t_refs[sid]
            err_ms = abs(pred_t - gt_t) * 1000
            if err_ms <= args.window_ms:
                tp += 1
                errors_ms.append(err_ms)
                matched_gt.add(sid)
            else:
                fp += 1
        else:
            fp += 1

    fn = len(gt_slides) - len(matched_gt)

    import numpy as np
    print("\n=== Evaluation Results ===")
    print(f"  Slides annotated (GT): {len(gt_slides)}")
    print(f"  Triggers fired:        {len(trigger_log)}")
    print(f"  TP (within {args.window_ms:.0f}ms):    {tp}")
    print(f"  FP:                    {fp}")
    print(f"  FN:                    {fn}")
    if errors_ms:
        print(f"  MAE:                   {np.mean(errors_ms):.1f}ms")
        print(f"  Max error:             {np.max(errors_ms):.1f}ms")
        print(f"  % within window:       {tp / len(gt_slides):.1%}")


if __name__ == "__main__":
    # Allow running as: python -m ppsync.cli preprocess/align/eval
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preprocess":
        preprocess_main(sys.argv[2:])
    elif cmd == "align":
        align_main(sys.argv[2:])
    elif cmd == "eval":
        eval_main(sys.argv[2:])
    else:
        print("Usage: python -m ppsync.cli [preprocess|align|eval] ...")
        sys.exit(1)
