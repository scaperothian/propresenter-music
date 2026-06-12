# propresenter-music

Real-time music alignment for live ProPresenter slide advancement.
(Python package and CLI prefix: `ppsync`.)

Given a studio reference recording and a slide manifest (slide IDs + timestamps), `ppsync` listens to live audio (microphone or house feed), tracks where the song is, and advances ProPresenter slides at the right moments via its REST API — even when playback starts mid-song, and through PA/room/mic coloration.

## How it works

```
OFFLINE (once per song)
  Reference audio + manifest JSON
    → MERT embeddings: one forward pass per sliding 2s window (100ms stride)
    → contrastive normalization (subtract song mean, L2)
    → per-slide prototype embeddings + HMM transition matrix
    → cached .npz

LIVE (per 200ms chunk)
  Audio ring buffer (last 2s) → one MERT forward → pooled embedding
    normalized against a blend of the song mean and the live stream's own
    running mean (cancels PA/mic coloration)

  Matcher  rigid 1:1 correlation against the reference (default — playback
           doesn't warp time), or subsequence DTW (live-band mode)
  Anchor   initial lock needs agreeing frames + a clear cost margin over the
           runner-up candidate (repeated choruses tie otherwise); big jumps
           must beat a local re-alignment by the same margin
  HMM      forward filter, fallback position during low-confidence stretches
  Trigger  fires the slide containing the position (catch-up on mid-song
           join), schedules a timer at the predicted boundary crossing, and
           drives ProPresenter via propresenter-client go_to_slide()
```

Measured on the Drive test set (studio reference, EQ'd/attenuated "mic-proxy" replay): lock-on ~5.6s after music is audible from any start offset, 14/14 slides fired at −400ms ±5ms from their boundaries, tracking error 0.20s median, ~85ms mean (107ms max) processing per 200ms chunk on Apple Silicon.

## Requirements

- Python 3.11+
- ~370MB for the MERT model (downloaded automatically on first run)
- Apple Silicon (MPS), CUDA, or CPU
- ProPresenter 7 with the network API enabled (Preferences → Network) for live triggering
- [propresenter-client](../propresenter-client) (installed as a path dependency) for slide control

## Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -e ../propresenter-client   # for --pp-host triggering
```

## Quickstart (real song)

### 1. Convert a ProPresenter annotation to a manifest

```bash
.venv/bin/python tools/pp_to_manifest.py studio_drive.json --artist "Incubus"
# → data/incubus/drive/incubus_drive_manifest.json
```

This flattens grouped slides with repeated trigger times into chronological slide *instances*, recording each instance's ProPresenter slide index (`pp_slide_index`) and the presentation UUID (`pp_uuid`) so repeated choruses trigger the correct presentation slide.

ProPresenter annotations carry the song title but not the artist, so `--artist` is required and written into the manifest. **Every per-song artifact lives in `data/<artist>/<song>/` and is named by the `<artist>_<title>` slug** (here `data/incubus/drive/` and `incubus_drive`) — manifest, embedding cache, benchmark results, telemetry — so when you process a second song (say Forrest Frank's "Good Day" → `data/forrest_frank/good_day/forrest_frank_good_day_*`), nothing collides and every file says which song it belongs to. The `data/` tree is gitignored: caches are rebuildable and manifests come from your annotation source.

### 2. Preprocess (offline, once per song)

```bash
.venv/bin/ppsync-preprocess data/incubus/drive/incubus_drive_manifest.json
# → data/incubus/drive/incubus_drive_cache.npz  (default: <artist>_<title>_cache.npz)
```

Live matching works by comparing the last 2s of
mic audio against *every* 2s window of the reference song. The npz files are 
embeddings for a sliding 2s window at every 50ms step of the song (thousands 
of forward passes — minutes of GPU work), and the cache also stores the 
song-level mean embedding (used to normalize live audio so "sounds like 
music" similarity doesn't swamp section differences), per-slide prototype 
embeddings for coarse matching, and the HMM transition matrix derived from slide durations. Live audio must be embedded *exactly* like the reference ones (same length, same MERT layer, same precision), so changing those settings means rebuilding the cache.

### 3. Live alignment → ProPresenter

```bash
# terminal 1: aligner
.venv/bin/ppsync-align data/incubus/drive/incubus_drive_cache.npz --mic \
    --pp-host localhost --pp-activate --log /tmp/ppsync_incubus_drive.jsonl

# terminal 2: live monitor dashboard (http://localhost:8765)
.venv/bin/python webapp/server.py --log /tmp/ppsync_incubus_drive.jsonl
```

`--pp-activate` switches ProPresenter to the cache's presentation if a different one is focused. Use `--dry-run` to print triggers without touching ProPresenter, and `--trigger-buffer MS` to tune how early slides land (default 200ms before the boundary; the pipeline adds ~200ms more earliness on top).

### 4. Benchmark against files (no mic needed)

```bash
.venv/bin/python tools/benchmark.py data/incubus/drive/incubus_drive_cache.npz \
    --file studio_drive.wav --manifest data/incubus/drive/incubus_drive_manifest.json \
    --offsets 0,30,62,90 [--duration 30] [--matcher dtw|rigid] \
    [--json-out data/incubus/drive/bench_incubus_drive_<experiment>.json]
```

Replays the file from each start offset and prints a per-slide report (target time, fired time, verdict) plus tracking error and per-chunk latency against the real-time budget. `--duration` tests partial-song slices; `--trace-out` dumps per-frame telemetry for debugging.

### 5. Verify ProPresenter triggering end to end

```bash
.venv/bin/python tools/pp_trigger_test.py data/incubus/drive/incubus_drive_manifest.json
```

Closed-loop: commands every distinct presentation slide, reads back the active slide index, restores the original slide. (Also available as auto-skipping pytests in `tests/test_pp_live.py`.)

## Manifest format

```json
{
  "song_id": "Drive",
  "artist": "Incubus",
  "ref_audio": "/path/to/studio_drive.wav",
  "pp_uuid": "C4F878BA-60EF-40CF-9500-7124FC891C87",
  "slides": [
    {"slide_id": "00_verse1", "t_ref": 0.0,    "lyrics": "...", "pp_slide_index": 0},
    {"slide_id": "04_chorus", "t_ref": 63.849, "lyrics": "...", "pp_slide_index": 4},
    {"slide_id": "10_chorus", "t_ref": 128.285, "lyrics": "...", "pp_slide_index": 4}
  ]
}
```

`t_ref` is the true musical timestamp in the reference audio. Repeated sections are separate instances sharing one `pp_slide_index`. `pp_uuid`/`pp_slide_index` are optional for dry-run/benchmark use.

`artist` + `song_id` form the song's filename slug (`incubus_drive`); the slug is stored in the embedding cache and stamped into telemetry logs and benchmark JSON, so every artifact identifies its song even outside its filename.

## CLI reference

### `ppsync-preprocess`

```
ppsync-preprocess <manifest.json> [options]

  --output FILE      .npz output path (default: <artist>_<title>_cache.npz
                     beside the manifest, i.e. in the song's data/ directory)
  --lookback SEC     embedding window length        (default: 2.0s)
  --stride SEC       reference window stride        (default: 0.1, validated
                     equal to 0.05 at half the preprocessing time)
  --batch-size N     windows per MERT forward       (default: 16; no speedup
                     beyond 16 on MPS, may help on CUDA)
  --layer N          MERT transformer layer         (default: 7)
  --device DEVICE    cpu | cuda | mps               (auto-detected)
```

The cache records the embedding precision (`MERT_FP16`) — rebuild after changing it.

### `ppsync-align`

```
ppsync-align <cache.npz> --mic | --file AUDIO | --list-devices [options]

  --input-device DEV    mic device index or name substring
  --start-offset SEC    skip N seconds of input file
  --no-realtime         process file faster than real time
  --pp-host HOST        trigger ProPresenter via REST (port --pp-port, default 1025)
  --pp-activate         activate the cache's presentation at startup
  --trigger-buffer MS   fire this early before each boundary (default: 200)
  --trigger-conf FLOAT  minimum confidence to fire (default: 0.6)
  --log FILE            JSON-lines telemetry (feeds webapp/server.py)
  --dry-run             print triggers, no HTTP
  --rest-url URL        legacy JSON POST endpoint (when --pp-host not set)
```

## Telemetry & monitor

Every chunk logs one JSON line (`--log`): matcher position/confidence/margin, anchor state (`initialized`, `obs_accepted`, `jump_pending`), HMM state, trigger events with exact fire times, processing latency, and trigger delivery results. `webapp/server.py` tails the log and serves a live dashboard (last triggered slide, confidence sparklines, song position, latency vs budget).

## Running tests

```bash
.venv/bin/pytest      # 55 tests, ~5s, no model download
                      # tests/test_pp_live.py runs only when ProPresenter
                      # is reachable on localhost:1025 (else auto-skips)
```

## Project layout

```
src/ppsync/
  config.py          all tunable constants (matcher, thresholds, fp16, ...)
  io.py              manifest + audio loading
  embed.py           MERT loading (layer-truncated, fp16) and inference
  windows.py         strided window pooling
  transform.py       contrastive normalization
  preprocess.py      offline sliding-window embedding cache builder
  dtw.py             rigid matcher (default) + subsequence DTW (live-band)
  hmm.py             online HMM forward filter
  audio_capture.py   mic (native-rate, backpressure-draining) + file sources
  aligner.py         pipeline: embedding → matcher → anchor → HMM → trigger
  trigger.py         scheduled trigger fires → ProPresenter / legacy POST
  telemetry.py       JSON-lines logger
  cli.py             CLI entry points

tools/
  pp_to_manifest.py        ProPresenter JSON → ppsync manifest
  benchmark.py             start-offset re-sync benchmark (per-slide reports)
  pp_trigger_test.py       closed-loop live ProPresenter trigger test
  generate_test_audio.py   synthetic test data
  diag_embed.py            embedding-consistency diagnostics
  diag_phase.py            chunk-phase sensitivity diagnostics

webapp/
  server.py          SSE server tailing the telemetry log
  index.html         live dashboard (slide box + sparklines)

data/                       (gitignored — local song artifacts)
  <artist>/<song>/          one directory per song, e.g. data/incubus/drive/
    <slug>_manifest.json    manifest (slug = <artist>_<song>)
    <slug>_cache.npz        embedding cache
    bench_<slug>_*.json     benchmark results history
```

## Design notes

See [CLAUDE.md](CLAUDE.md) for the non-obvious design decisions and their
experimental evidence: why live and reference embeddings must come from the
same computation, why the rigid matcher replaced DTW for playback, the
repeat-ambiguity guards (margin-gated lock, jump guard), live-mean
adaptation for mic coloration, and the scheduled-fire trigger model.
