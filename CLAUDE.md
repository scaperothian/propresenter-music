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

embed_audio(wav) → [L+1, T, D] hidden states
hidden[layer]    → [T, D] frame embeddings

strided_window_embeddings(frames, lookback=2s, stride=20ms)
  → raw_win_embs [N_ref, D]  +  ref_timestamps [N_ref]

fit_global(raw_win_embs)         → global_emb [D]
apply_contrastive(raw, global)   → ref_embs [N_ref, D]  (normalized)

pool_slide_embeddings per slide  → slide_protos [N_slides, D]
build_hmm_transition(t_refs, t_stops, stride_sec) → A [N,N], pi [N]

── saved to .npz ──

Live (per 200ms chunk):
  embed_chunk_live → frames [T_chunk, D]
  ring buffer of frames → mean pool → pooled_raw [D]
  apply_contrastive(pooled_raw, global_emb) → pooled_norm [D]

  cosine(pooled_norm, slide_protos) → coarse_slide_idx, coarse_conf

  dtw.align(live_buffer, ref_embs, ref_timestamps, search_bounds)
    → Step 1: similarity_search → candidate_t
    → Step 2: subsequence_dtw  → refined_t, path_cost, confidence

  HMMPredictor.update(refined_t, dtw_confidence, coarse_slide_idx)
    → current_slide, state_probs, predicted_next_t, trigger_confidence

  TriggerScheduler.update(...)
    → HTTP POST if confidence high and near boundary
```

## Non-obvious design decisions

**No Sakoe-Chiba band in `subsequence_dtw`.**  A band of `|i - j| <= k` is wrong for subsequence DTW because the optimal path is offset by the match position, not near (0,0).  The reference window passed by `align()` is already narrow (±`dtw_context_sec` around the candidate), which limits the search space without breaking correctness.

**Contrastive normalization, not ZCA.**  `apply_contrastive` subtracts the song-level mean then L2-normalizes.  This removes the dominant "sounds like music" direction that makes all sections score ~0.9 cosine similarity.  ZCA from `mert-experiment` is more powerful but expensive; add it if per-slide similarity remains too high.

**HMM transition from stride_sec, not chunk_sec.**  The HMM step interval is `stride_sec` (20ms), matching the reference embedding density.  This means the HMM advances once per reference frame, not once per audio chunk.  The `SongAligner` calls `hmm.update()` once per audio chunk (200ms), so the HMM effectively sees observations at 5Hz, not 50Hz — this is fine since the reference timestamps step at 20ms but the live update rate is 200ms.

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
