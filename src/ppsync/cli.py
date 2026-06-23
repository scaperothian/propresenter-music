"""CLI entry points for ppsync.

Commands
--------
ppsync-preprocess   Offline preprocessing: build reference embedding cache.
ppsync-align        Live alignment (mic or file).
ppsync-eval         Offline evaluation against ground-truth annotations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    CHUNK_SEC,
    DTW_LIVE_SEC,
    DTW_SEARCH_SEC,
    EMBED_BATCH_SIZE,
    LOOKBACK_SEC,
    MERT_LAYER,
    REST_URL,
    STRIDE_SEC,
    TARGET_SR,
    TRIGGER_BUFFER_MS,
    TRIGGER_CONFIDENCE_MIN,
)
from .logconf import configure_logging

log = logging.getLogger(__name__)


def _add_logging_args(p: argparse.ArgumentParser) -> None:
    """Add the shared --log-level / -v verbosity flags to a parser."""
    p.add_argument(
        "--log-level", default="info", metavar="LEVEL",
        help="Logging verbosity: debug | info | warning | error (default: info). "
             "Logs are written to stdout.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Shortcut for --log-level debug (shows the per-chunk status line).",
    )


def _resolve_level(args: argparse.Namespace) -> str:
    """Verbose flag wins; otherwise the explicit --log-level."""
    return "debug" if getattr(args, "verbose", False) else args.log_level


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
        help="Output .npz path (default: <artist>_<title>_cache.npz beside the "
             "JSON, e.g. incubus_drive_cache.npz).",
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
    p.add_argument(
        "--batch-size", type=int, default=EMBED_BATCH_SIZE, metavar="N",
        help=f"Windows per MERT forward pass (default: {EMBED_BATCH_SIZE}).  "
             "Larger = faster preprocessing while GPU memory allows.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    _add_logging_args(p)
    return p


def preprocess_main(argv: list[str] | None = None) -> None:
    from .io import load_song_meta
    from .preprocess import preprocess_song

    args = _preprocess_parser().parse_args(argv)
    level = _resolve_level(args)
    if args.quiet and level == "info":
        level = "warning"  # --quiet drops progress unless level set explicitly
    configure_logging(level)
    manifest_path = Path(args.manifest)
    if args.output:
        output_path = Path(args.output)
    else:
        # Name the cache by artist+title so caches for different songs are
        # identifiable from the filename alone.
        slug = load_song_meta(manifest_path)["slug"]
        output_path = manifest_path.parent / f"{slug}_cache.npz"
    preprocess_song(
        manifest_path=manifest_path,
        output_path=output_path,
        lookback_sec=args.lookback,
        stride_sec=args.stride,
        mert_layer=args.layer,
        device=args.device,
        show_progress=not args.quiet,
        batch_size=args.batch_size,
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
    p.add_argument(
        "cache", nargs="?", default=None,
        help="Path to .npz cache from ppsync-preprocess.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="Capture from system microphone.")
    src.add_argument("--file", metavar="AUDIO", help="Simulate real-time from a file.")
    src.add_argument(
        "--list-devices", action="store_true",
        help="List audio input devices and exit (no cache needed).",
    )

    p.add_argument(
        "--input-device", default=None, metavar="DEV",
        help="Input device index or name substring for --mic (default: system input).",
    )

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
        help=f"Legacy POST endpoint for slide triggers (default: {REST_URL}).",
    )
    p.add_argument(
        "--pp-host", default=None, metavar="HOST",
        help="ProPresenter host — triggers slides via propresenter-client "
             "go_to_slide() against the active presentation.",
    )
    p.add_argument(
        "--pp-port", type=int, default=1025, metavar="PORT",
        help="ProPresenter API port (default: 1025).",
    )
    p.add_argument(
        "--pp-activate", action="store_true",
        help="Activate the cache's presentation in ProPresenter at startup "
             "if it is not already the active one.",
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
    _add_logging_args(p)
    return p


def align_main(argv: list[str] | None = None) -> None:
    import torch

    from .aligner import SongAligner
    from .audio_capture import FileCapture, MicCapture
    from .embed import load_model
    from .telemetry import TelemetryLogger

    args = _align_parser().parse_args(argv)
    configure_logging(_resolve_level(args))

    if args.list_devices:
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                default = " (default)" if idx == sd.default.device[0] else ""
                print(f"  [{idx}] {dev['name']}  "
                      f"{int(dev['default_samplerate'])} Hz{default}")
        return

    if not args.cache:
        _align_parser().error("cache is required unless --list-devices is given")

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
    pp_controller = None
    if args.pp_host:
        from propresenter_client.main import ProPresenterController

        pp_controller = ProPresenterController(host=args.pp_host, port=args.pp_port)
        if pp_controller.get_status() is None:
            print(f"Error: cannot reach ProPresenter at "
                  f"{args.pp_host}:{args.pp_port} — is the network API enabled?")
            sys.exit(1)
        print(f"ProPresenter connected at {args.pp_host}:{args.pp_port}")

    aligner = SongAligner(
        cache_path=cache_path,
        model=model,
        processor=processor,
        device=device,
        rest_url=args.rest_url,
        pp_controller=pp_controller,
        trigger_buffer_ms=args.trigger_buffer,
        trigger_conf_min=args.trigger_conf,
        dry_run=args.dry_run,
        dtw_live_sec=args.dtw_live,
        dtw_search_sec=args.dtw_search,
        chunk_sec=args.chunk,
    )

    artist_str = f"{aligner.artist} — " if aligner.artist else ""
    print(f"Song: {artist_str}{aligner.song_id}  [{aligner.song_slug}]")

    # The trigger drives the ACTIVE presentation (go_to_slide), so make sure
    # the right one is focused — slide indices are meaningless otherwise.
    if pp_controller is not None and aligner.pp_uuid:
        active_uuid = pp_controller.get_active_presentation_uuid()
        if active_uuid != aligner.pp_uuid:
            if args.pp_activate:
                print(f"Activating presentation {aligner.pp_uuid}…")
                if not pp_controller.activate_presentation(aligner.pp_uuid):
                    print("Error: could not activate the presentation.")
                    sys.exit(1)
            else:
                print(f"WARNING: active presentation is {active_uuid}, but this "
                      f"cache belongs to {aligner.pp_uuid}.\n"
                      f"         Triggers would hit the wrong slides — focus the "
                      f"right presentation or pass --pp-activate.")

    if args.mic:
        input_dev = args.input_device
        if input_dev is not None and input_dev.lstrip("-").isdigit():
            input_dev = int(input_dev)
        source = MicCapture(chunk_sec=args.chunk, device=input_dev)
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

    with TelemetryLogger(log_path) as telemetry:
        # First log line identifies the session's song so a telemetry file is
        # self-describing (the webapp shows it; offline tooling can filter).
        telemetry.log({
            "event": "meta",
            "song_id": aligner.song_id,
            "artist": aligner.artist,
            "song_slug": aligner.song_slug,
            "cache": str(cache_path),
        })
        try:
            for chunk, wall_t in source:
                frame = aligner.process_chunk(chunk, chunk_wall_t=wall_t)
                telemetry.log(frame)
                if frame.get("status"):  # buffering / silence — no telemetry row
                    continue
                # Per-chunk status is the hot-path line; gate it so the f-string
                # is not even built (let alone written) unless DEBUG is on.
                if log.isEnabledFor(logging.DEBUG):
                    status = (
                        f"  chunk={frame['chunk']:4d}"
                        f"  dtw_t={frame['dtw_refined_t']:6.2f}s"
                        f"  dtw_conf={frame['dtw_confidence']:.2f}"
                        f"  slide=[{frame['hmm_current_slide_id']}]"
                        f"  trigger_conf={frame['hmm_trigger_confidence']:.2f}"
                    )
                    if frame["triggered"]:
                        status += f"  *** TRIGGER → {frame['triggered_slide_id']} ***"
                    log.debug(status)
        except KeyboardInterrupt:
            log.info("Stopped.")


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
    _add_logging_args(p)
    return p


def eval_main(argv: list[str] | None = None) -> None:
    import torch

    from .aligner import SongAligner
    from .audio_capture import FileCapture
    from .embed import load_model
    from .io import finalize_slide_stops, load_audio, load_manifest
    from .telemetry import TelemetryLogger

    args = _eval_parser().parse_args(argv)
    configure_logging(_resolve_level(args))

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

    with TelemetryLogger(log_path) as telemetry:
        for chunk, wall_t in source:
            frame = aligner.process_chunk(chunk, chunk_wall_t=wall_t)
            telemetry.log(frame)
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
