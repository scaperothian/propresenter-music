# CLAUDE.md — ppsync

Real-time music alignment: MERT embeddings → subsequence DTW → HMM → REST trigger.

## Commands

```bash
# Install
python3.11 -m venv .venv && .venv/bin/pip install -e .

# Tests (no model download, ~1s)
.venv/bin/pytest
.venv/bin/pytest -v

# Generate synthetic test audio
.venv/bin/python tools/generate_test_audio.py

# Preprocess a song (downloads MERT ~370MB on first run)
.venv/bin/ppsync-preprocess data/test_manifest.json --output data/test_cache.npz

# Align from file
.venv/bin/ppsync-align data/test_cache.npz --file data/test_song.wav --dry-run

# Align from mic
.venv/bin/ppsync-align data/test_cache.npz --mic --dry-run

# Evaluate accuracy
.venv/bin/ppsync-eval data/test_cache.npz \
    --file data/test_song.wav \
    --ground-truth data/test_manifest.json

# Convert a ProPresenter annotation JSON to a ppsync manifest
.venv/bin/python tools/pp_to_manifest.py <song>.json -o <song>_manifest.json

# Live mic -> ProPresenter (REST API, default port 1025)
.venv/bin/ppsync-align data/studio_cache_sliding.npz --mic \
    --pp-host localhost [--pp-port 1025] [--log /tmp/ppsync.jsonl]

# Start-offset re-sync benchmark (file-based, no mic; see tools/benchmark.py)
.venv/bin/python tools/benchmark.py data/studio_cache_sliding.npz \
    --file <song>.wav --manifest <song>_manifest.json \
    --offsets 0,30,64.1,95 [--duration 30] [--trace-out /tmp/trace.json]
```

## Package layout

| Module | Key exports |
|---|---|
| `config.py` | All tunable constants — change defaults here |
| `io.py` | `load_manifest`, `load_audio`, `finalize_slide_stops` |
| `embed.py` | `load_model`, `embed_audio` (offline full song), `embed_chunk_live` (streaming) |
| `windows.py` | `strided_window_embeddings`, `pool_slide_embeddings` |
| `transform.py` | `fit_global`, `apply_contrastive` (subtract global + L2-norm) |
| `preprocess.py` | `preprocess_song`, `load_cache`, `build_hmm_transition` |
| `dtw.py` | `cosine_distance_matrix`, `subsequence_dtw`, `similarity_search`, `align` |
| `hmm.py` | `HMMPredictor` — online forward filter |
| `audio_capture.py` | `MicCapture`, `FileCapture` |
| `aligner.py` | `SongAligner` — wires all three layers together |
| `trigger.py` | `TriggerScheduler` |
| `telemetry.py` | `TelemetryLogger` |
| `cli.py` | `preprocess_main`, `align_main`, `eval_main` |

## Data flow

```
JSON manifest → load_manifest → slides (slide_id, t_ref, t_stop)
Audio file   → load_audio    → [N] float32 @ 24kHz

sliding_window_embeddings(wav, lookback=2s, stride=50ms)
  one MERT forward per [t-2s, t] window (batched), mean-pooled
  → raw_win_embs [N_ref, D]  +  ref_timestamps [N_ref]

fit_global(raw_win_embs)         → global_emb [D]
apply_contrastive(raw, global)   → ref_embs [N_ref, D]  (normalized)

pool_slide_embeddings per slide  → slide_protos [N_slides, D]
build_hmm_transition(t_refs, t_stops, stride_sec) → A [N,N], pi [N]

── saved to .npz ──

Live (per 200ms chunk):
  audio ring buffer (last 2s) → embed_chunk_live(whole window) → mean pool
  → pooled_raw [D]   (buffering until ring + DTW buffer are full, ~6s)
  apply_contrastive(pooled_raw, global_emb) → pooled_norm [D]

  cosine(pooled_norm, slide_protos) → coarse_slide_idx, coarse_conf

  dtw.align(live_buffer, ref_embs, ref_timestamps, search_bounds)
    → Step 1: similarity_search → candidate_t
    → Step 2: subsequence_dtw  → refined_t, path_cost, confidence

  HMMPredictor.update(refined_t, dtw_confidence, coarse_slide_idx)
    → current_slide, state_probs, predicted_next_t, trigger_confidence

  pos_t = refined_t if DTW confident else HMM expected_pos_t
  first unfired boundary vs pos_t (± grace) → TriggerScheduler.update(...)
    → HTTP POST when pos_t crosses t_ref - buffer
```

## Non-obvious design decisions

**Live and reference embeddings must come from the same computation.**  MERT
frames depend on their attention context: frames from 30s chunks, 2s windows,
and 0.2s chunks live in *different distributions* and cosine matching across
them fails completely (best match lands at the song's quiet outro).  Both
sides therefore embed full 2s windows in single MERT calls.  Caching per-chunk
live frames is also out: it makes the embedding depend on chunk phase, and a
0.1s phase shift between live playback and the reference grid drops tracking
from 100% to 4% (see `tools/benchmark.py` history in data/bench_studio*.json).

**Buffer warm-up gating.**  No DTW, search-window anchoring, or HMM
observation until the audio ring spans the full lookback AND the DTW buffer is
full (~6s).  A 1-frame DTW query matches anywhere with above-threshold
"confidence", and the forward-only anchor then locks the search at a bogus
position permanently.

**Trigger fires on DTW position, not HMM expectation.**  The HMM's
`expected_pos_t` is a probability-weighted average of slide *midpoints* — it
crosses a boundary only after the boundary has passed, so boundary triggers
driven by it fire seconds late or never.  When DTW is confident the trigger
compares `refined_t` against the slide instance containing it (see
`select_trigger_boundary`); the HMM is the fallback during low-confidence
stretches.  On lock-on (mid-song join) or after a boundary is stepped over by
jitter, the CURRENT slide fires immediately as a catch-up — the screen must
show where the song is now — and only instances strictly before it are
skipped.

**Mic capture: native rate + queue draining + silence gate.**  `MicCapture`
opens the stream at the device's native sample rate and resamples each block
(devices often refuse 24kHz).  If alignment falls behind, all queued blocks
are drained into one combined chunk — the audio ring absorbs any chunk size —
so latency stays bounded instead of growing forever.  The aligner skips all
layers while the lookback RMS is under `SILENCE_RMS_DBFS`: ambient noise
DTW-matches the song's quietest section with above-threshold confidence, so
an open mic before the song starts could otherwise walk into a boundary.

**Manifest slide instances vs ProPresenter slide indices.**  ppsync slides are
chronological *trigger events* (a chorus shown 3× = 3 instances), but the
ProPresenter REST API addresses slides by their position in the presentation.
`pp_to_manifest.py` records `pp_slide_index` per instance (repeats share one)
and the presentation `pp_uuid`; both ride through the cache, and the trigger
fires `GET /v1/presentation/{uuid}/{pp_slide_index}/trigger` (`active` when no
uuid).  Trigger HTTP runs on a daemon thread — a slow ProPresenter must not
stall the 200ms audio loop.  Enable with `--pp-host` (else legacy POST).

**No Sakoe-Chiba band in `subsequence_dtw`.**  A band of `|i - j| <= k` is wrong for subsequence DTW because the optimal path is offset by the match position, not near (0,0).  The reference window passed by `align()` is already narrow (±`dtw_context_sec` around the candidate), which limits the search space without breaking correctness.

**Contrastive normalization, not ZCA.**  `apply_contrastive` subtracts the song-level mean then L2-normalizes.  This removes the dominant "sounds like music" direction that makes all sections score ~0.9 cosine similarity.  ZCA from `mert-experiment` is more powerful but expensive; add it if per-slide similarity remains too high.

**HMM transition step mismatch (known, tolerated).**  The transition matrix is built for `stride_sec` steps but `hmm.update()` runs once per 200ms chunk, so the transition prior advances slower than real time.  With confident DTW observations the emission dominates and this barely matters; it is why the HMM alone cannot drive timely triggers (see trigger note above).  Fix by rebuilding A for the chunk interval if the HMM ever needs to free-run through long low-confidence gaps.

**Search window anchoring.**  Once `dtw_confidence >= CONFIDENCE_THRESHOLD`, the lower bound of the cosine search window advances to `refined_t - 2s`.  This prevents backward regression but allows the search to slip back 2s to absorb timing variation.

**Cold start.**  HMM starts with a uniform prior.  After the first coarse MERT match exceeds threshold, `set_prior_from_coarse()` concentrates belief on the detected slide; then DTW+HMM take over.

## Test coverage

| File | What it covers |
|---|---|
| `test_dtw.py` | cosine distance, subsequence DTW (identical subsequence, confidence, empty), similarity search bounds, full `align()` return keys |
| `test_hmm.py` | transition matrix (row sums, left-to-right, absorbing last state), `update()` keys, state probs sum to 1, convergence, drift, reset, trigger confidence near boundary |
| `test_io.py` | manifest parsing (stops inferred, finalize, empty raises), audio resampling and stereo→mono |
| `test_transform.py` | global mean, L2 norm after contrastive, single vector, zero-out identical rows |
| `test_windows.py` | window count, timestamps monotone, short audio → empty, pool range, pool empty range |
