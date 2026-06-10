# ppsync

Real-time music alignment for live ProPresenter slide advancement.

Given a studio reference recording and a slide manifest (slide IDs + timestamps), `ppsync` listens to live audio (microphone or house feed) and fires timed REST calls to advance ProPresenter slides — even when playback starts mid-song.

## How it works

```
OFFLINE (once per song)
  Reference audio + manifest JSON
    → MERT embeddings (dense 2s windows, 20ms stride)
    → per-slide prototype embeddings
    → HMM transition matrix
    → cached .npz file

LIVE (per 200ms chunk)
  Audio chunk (mic / file)
    → Layer 1: MERT coarse alignment   cosine search against slide prototypes
    → Layer 2: Subsequence DTW         refine position in reference sequence
    → Layer 3: HMM forward filter      smooth state, predict next boundary
    → Trigger scheduler                REST POST 200ms before boundary
```

The system handles cold starts by maintaining a uniform position prior until the MERT coarse match exceeds a confidence threshold, at which point the HMM seeds itself and the DTW search window locks forward.

## Requirements

- Python 3.11+
- ~370MB for the MERT model (downloaded automatically on first run via HuggingFace)
- Apple Silicon (MPS), CUDA, or CPU

## Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

## Quickstart

### 1. Generate synthetic test data

No real audio needed to try the pipeline:

```bash
.venv/bin/python tools/generate_test_audio.py
# → data/test_song.wav   (8 sections, distinct tones)
# → data/test_manifest.json
```

### 2. Preprocess (offline, once per song)

```bash
.venv/bin/ppsync-preprocess data/test_manifest.json --output data/test_cache.npz
```

Downloads MERT on first run, then runs the full audio through it and saves the embedding cache. Takes roughly 2–5× the song duration on CPU.

### 3. Run alignment

From a file (simulates real-time):

```bash
.venv/bin/ppsync-align data/test_cache.npz --file data/test_song.wav --dry-run
```

From the microphone:

```bash
.venv/bin/ppsync-align data/test_cache.npz --mic --dry-run
```

`--dry-run` prints trigger events without making HTTP requests. Remove it (and set `--rest-url`) to send real slide commands.

Test cold-start sync by skipping into the middle of the song:

```bash
.venv/bin/ppsync-align data/test_cache.npz --file data/test_song.wav \
    --start-offset 35 --dry-run
```

### 4. Evaluate accuracy

```bash
.venv/bin/ppsync-eval data/test_cache.npz \
    --file data/test_song.wav \
    --ground-truth data/test_manifest.json \
    --window-ms 500
```

Reports MAE (ms), TP/FP/FN counts, and percentage of triggers within the tolerance window.

## Manifest format

```json
{
  "song_id": "amazing_grace",
  "ref_audio": "amazing_grace.wav",
  "slides": [
    {"slide_id": "intro",        "t_ref": 0.0,  "lyrics": ""},
    {"slide_id": "verse1_line1", "t_ref": 8.5,  "lyrics": "Amazing grace, how sweet the sound"},
    {"slide_id": "verse1_line2", "t_ref": 12.8, "lyrics": "That saved a wretch like me"}
  ]
}
```

`t_ref` is the TRUE musical timestamp in the reference audio (seconds). `ref_audio` resolves relative to the JSON file.

## CLI reference

### `ppsync-preprocess`

```
ppsync-preprocess <manifest.json> [options]

  --output FILE      .npz output path (default: <manifest>.npz)
  --lookback SEC     embedding window lookback   (default: 2.0s)
  --stride  SEC      window stride               (default: 0.020s = 20ms)
  --layer   N        MERT transformer layer 0-12 (default: 7)
  --device  DEVICE   cpu | cuda | mps            (auto-detected)
  --quiet            suppress tqdm progress
```

### `ppsync-align`

```
ppsync-align <cache.npz> --mic | --file AUDIO [options]

  --start-offset SEC    skip N seconds of input file
  --no-realtime         process file faster than real time
  --rest-url URL        slide trigger endpoint (default: http://localhost:5000/slide)
  --trigger-buffer MS   lead time before boundary (default: 200ms)
  --trigger-conf FLOAT  minimum HMM confidence to fire (default: 0.6)
  --chunk SEC           audio chunk size (default: 0.2s)
  --dtw-live SEC        live buffer sent to DTW (default: 4.0s)
  --dtw-search SEC      forward search window in reference (default: 45s)
  --log FILE            JSON-lines telemetry output
  --dry-run             log triggers, skip HTTP
  --device DEVICE       cpu | cuda | mps
```

### `ppsync-eval`

```
ppsync-eval <cache.npz> --file AUDIO --ground-truth JSON [options]

  --start-offset SEC    skip N seconds
  --log FILE            JSON-lines telemetry
  --window-ms MS        tolerance for TP (default: 500ms)
  --device DEVICE
```

## REST trigger payload

Each trigger fires an HTTP POST:

```json
{
  "slide_id": "verse1_line2",
  "slide_idx": 2,
  "timestamp": 12.8,
  "confidence": 0.83,
  "current_t": 12.61
}
```

## Telemetry log

With `--log out.jsonl`, every 200ms chunk emits one JSON line:

```json
{
  "chunk": 42,
  "coarse_slide_id": "verse1_line1", "coarse_confidence": 0.71,
  "dtw_refined_t": 10.24, "dtw_confidence": 0.78,
  "hmm_current_slide_id": "verse1_line1", "hmm_predicted_next_t": 12.80,
  "hmm_trigger_confidence": 0.61,
  "triggered": false,
  "processing_ms": 187.3
}
```

## Running tests

```bash
.venv/bin/pytest            # 33 tests, ~1s, no model download
.venv/bin/pytest -v         # verbose
```

## Project layout

```
src/ppsync/
  config.py          default constants
  io.py              manifest + audio loading
  embed.py           MERT model and inference
  windows.py         strided window pooling
  transform.py       contrastive normalization
  preprocess.py      offline embedding cache builder
  dtw.py             subsequence DTW alignment
  hmm.py             online HMM forward filter
  audio_capture.py   mic and file audio sources
  aligner.py         three-layer pipeline
  trigger.py         REST trigger scheduler
  telemetry.py       JSON-lines logger
  cli.py             CLI entry points

tools/
  generate_test_audio.py   synthetic test data generator

data/
  example_manifest.json    example manifest (Amazing Grace)

tests/
  test_dtw.py
  test_hmm.py
  test_io.py
  test_transform.py
  test_windows.py
```
