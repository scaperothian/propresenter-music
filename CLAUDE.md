# CLAUDE.md — propresenter-music (package: ppsync)

Real-time music alignment: MERT embeddings → rigid/DTW matcher → HMM → scheduled ProPresenter trigger.

## Workflow

**Never commit directly to `main`.**  All changes go on a feature branch and
land via a GitHub pull request (`gh pr create` against
`scaperothian/propresenter-music`).  Branch from up-to-date `main`, push the
branch, open the PR, and merge only after review.  Tags like `v0.1.0` mark
regression baselines — see the tag annotation for verified benchmark numbers
and the repro command.

## Commands

```bash
# Install
python3.11 -m venv .venv && .venv/bin/pip install -e .

# Tests (no model download, ~5s)
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
# (default output: data/<artist>/<title>/<artist>_<title>_manifest.json)
.venv/bin/python tools/pp_to_manifest.py <song>.json --artist "<Artist>"

# Live mic -> ProPresenter (REST API, default port 1025)
.venv/bin/ppsync-align data/incubus/drive/incubus_drive_cache.npz --mic \
    --pp-host localhost [--pp-activate] [--trigger-buffer 0] \
    [--log /tmp/ppsync_incubus_drive.jsonl]

# Live monitor web UI (tails the --log file; see webapp/README.md)
.venv/bin/python webapp/server.py \
    --log /tmp/ppsync_incubus_drive.jsonl   # localhost:8765

# Offline analysis web UI (plots benchmark --trace-out files; localhost:8765/analysis)
# overlays rigid vs DTW position-vs-time so the stalling effect is visible
.venv/bin/python webapp/server.py --trace-dir /tmp/ppsync_traces

# Start-offset re-sync benchmark (file-based, no mic; see tools/benchmark.py)
.venv/bin/python tools/benchmark.py data/incubus/drive/incubus_drive_cache.npz \
    --file <song>.wav --manifest <song>_manifest.json \
    --offsets 0,30,64.1,95 [--duration 30] [--matcher dtw|rigid] \
    [--trace-out /tmp/ppsync_traces/run.json]   # self-describing {meta,frames}

# Closed-loop ProPresenter trigger test (changes slides, restores after)
.venv/bin/python tools/pp_trigger_test.py data/incubus/drive/incubus_drive_manifest.json
```

## Package layout

| Module | Key exports |
|---|---|
| `config.py` | All tunable constants — change defaults here |
| `io.py` | `load_manifest`, `load_audio`, `finalize_slide_stops`, `song_slug`, `song_dir`, `load_song_meta` |
| `embed.py` | `load_model` (layer-truncated, fp16), `embed_audio`, `embed_chunk_live`, `prep_inputs` |
| `windows.py` | `strided_window_embeddings`, `pool_slide_embeddings` |
| `transform.py` | `fit_global`, `apply_contrastive` (subtract global + L2-norm) |
| `preprocess.py` | `preprocess_song`, `sliding_window_embeddings`, `load_cache`, `build_hmm_transition` |
| `dtw.py` | `rigid_align` (default matcher), `subsequence_dtw`, `align`, `topk_candidates`, `similarity_search` |
| `hmm.py` | `HMMPredictor` — online forward filter |
| `audio_capture.py` | `MicCapture` (native rate, drain), `FileCapture` |
| `aligner.py` | `SongAligner`, `select_trigger_boundary` |
| `trigger.py` | `TriggerScheduler` — scheduled (timer-based) fires, ProPresenter mode |
| `telemetry.py` | `TelemetryLogger` |
| `cli.py` | `preprocess_main`, `align_main`, `eval_main` |

## Data flow

```
JSON manifest → load_manifest → slides (slide_id, t_ref, t_stop)
Audio file   → load_audio    → [N] float32 @ 24kHz

sliding_window_embeddings(wav, lookback=2s, stride=100ms)
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

  matcher (MATCHER config / --matcher):
    rigid_align — 1:1 time mapping, mean cosine per offset  (default)
    align       — top-K cosine candidates + subsequence DTW (live-band)
    → refined_t, confidence, cost_margin

  anchor logic: initial lock (consistency + cost margin) / jump guard
    → obs_accepted, confirmed_t, search window

  HMMPredictor.update(refined_t, conf if obs_accepted else 0, coarse_idx)
    → current_slide, expected_pos_t, trigger_confidence  (fallback position)

  pos_t = refined_t if obs_accepted else HMM expected_pos_t
  select_trigger_boundary(pos_t) → catch-up / next boundary
  TriggerScheduler.update(...)
    → fire now if pos_t past (t_ref - buffer), else arm a timer at the
      predicted crossing (re-armed per estimate; go_to_slide / POST on
      a daemon thread)
```

## Non-obvious design decisions

**Per-song artifact naming (multi-song support).**  Manifests carry `artist`
+ `song_id`; `io.song_slug(artist, title)` builds the `<artist>_<title>` slug
(`incubus_drive`) and `io.song_dir(artist, title)` the per-song directory
(`data/<artist>/<song>/`, e.g. `data/incubus/drive/`).  EVERY per-song
artifact lives in that directory and is named with the slug: manifest
(`<slug>_manifest.json`, the `pp_to_manifest.py` default — `--artist` is
required because ProPresenter annotations store only the title), embedding
cache (`<slug>_cache.npz`, the `ppsync-preprocess` default, written beside
the manifest), benchmark results (`bench_<slug>_<experiment>.json` by
convention, with the song identity also embedded inside the JSON), and
telemetry logs (first line is a `{"event": "meta", ...}` record with
song/artist/slug; the webapp shows it in its header).  The whole `data/`
tree is gitignored — caches are rebuildable, manifests regenerate from the
annotation source.  The cache stores `song_id`/`artist`/`song_slug`; caches
from before this convention fall back to the cache filename stem.  Never
name a new song's files after the audio variant alone ("studio", "live") —
that is exactly the ambiguity the slug removes.

**Live and reference embeddings must come from the same computation.**  MERT
frames depend on their attention context: frames from 30s chunks, 2s windows,
and 0.2s chunks live in *different distributions* and cosine matching across
them fails completely (best match lands at the song's quiet outro).  Both
sides therefore embed full 2s windows in single MERT calls.  Caching per-chunk
live frames is also out: it makes the embedding depend on chunk phase, and a
0.1s phase shift between live playback and the reference grid drops tracking
from 100% to 4% (see `tools/benchmark.py` history in data/incubus/drive/bench_incubus_drive_studio*.json).

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
and the presentation `pp_uuid`; both ride through the cache.  Triggering goes
through `propresenter-client`'s `ProPresenterController.go_to_slide(n)`
(1-indexed, ACTIVE presentation); the CLI verifies the active presentation's
uuid against the cache at startup and `--pp-activate` switches to it.  Trigger
requests run on a daemon thread — a slow ProPresenter must not stall the
200ms audio loop.  Enable with `--pp-host`; closed-loop integration test:
`tools/pp_trigger_test.py`.

**Rigid matcher is the default; DTW is for the live-band mode.**  Playback of
a fixed recording does not warp time — only the acoustic channel differs — so
the matcher slides the live query across the reference with the time mapping
FIXED at 1:1 (`dtw.rigid_align`, `MATCHER="rigid"`, benchmark `--matcher`).
DTW's path flexibility absorbs acoustic mismatch by bending time, which shows
up as 0.5-1s position lag and wrong-repeat jumps under mic/PA coloration;
rigid matching made colored-audio results identical to clean-studio results
(14/14 fires at −400ms ±5ms, tracking 0.20s).  Keep DTW for genuinely
tempo-variable sources (live band).

**No Sakoe-Chiba band in `subsequence_dtw`.**  A band of `|i - j| <= k` is wrong for subsequence DTW because the optimal path is offset by the match position, not near (0,0).  The reference window passed by `align()` is already narrow (±`dtw_context_sec` around the candidate), which limits the search space without breaking correctness.

**Contrastive normalization, not ZCA.**  `apply_contrastive` subtracts the song-level mean then L2-normalizes.  This removes the dominant "sounds like music" direction that makes all sections score ~0.9 cosine similarity.  ZCA from `mert-experiment` is more powerful but expensive; add it if per-slide similarity remains too high.

**HMM transition step mismatch (known, tolerated).**  The transition matrix is built for `stride_sec` steps but `hmm.update()` runs once per 200ms chunk, so the transition prior advances slower than real time.  With confident DTW observations the emission dominates and this barely matters; it is why the HMM alone cannot drive timely triggers (see trigger note above).  Fix by rebuilding A for the chunk interval if the HMM ever needs to free-run through long low-confidence gaps.

**Search window anchoring.**  Once `dtw_confidence >= CONFIDENCE_THRESHOLD`, the lower bound of the cosine search window advances to `refined_t - 2s`.  This prevents backward regression but allows the search to slip back 2s to absorb timing variation.

**Initial lock and jump guard (repeat ambiguity).**  Riff-based songs make the
single best 2s cosine match unreliable — verse/outro/chorus repeats are
near-identical, and one wrong confident frame would ratchet the anchor to the
wrong repeat (observed live: lock onto the outro → slide 14 fires).  Defenses
(`config.py` INIT_*/JUMP_*): before the anchor exists, the cosine search
returns top-K separated candidates and DTW-refines each (lowest path cost
wins), and locking requires `INIT_CONSISTENT_FRAMES` consecutive confident
frames agreeing within `INIT_AGREE_SEC`; after lock, a forward jump larger
than `JUMP_GUARD_SEC` needs `JUMP_CONFIRM_FRAMES` agreeing frames before the
anchor, HMM, or trigger see it — AND the jump target must beat a local
re-alignment near the current anchor by `JUMP_MIN_COST_MARGIN` (a stable
wrong match agrees with itself; only a cost margin separates a real seek from
a wrong repeat).  Locking likewise requires a best-vs-runner-up margin
(`INIT_MIN_COST_MARGIN`) with a 16s pre-lock query buffer.  Triggers never
fire pre-lock.  The DTW query buffer holds `DTW_LIVE_SEC=6s` but starts
matching at `DTW_MIN_LIVE_SEC=4s`, so warm-up stays ~6s.

**Live-mean adaptation (mic/PA coloration).**  Live frames are contrastive-
normalized against a blend of the cache's song mean and the live stream's own
running mean (`LIVE_MEAN_ADAPT_SEC`).  Coloration shifts all live embeddings
by a common offset; without cancelling it the instrumental outro becomes
every frame's nearest neighbour (observed live AND on EQ'd test audio).
Trade-off: during the first ~20s the mean is immature and fires can lag ~1-2s
even on clean audio; colored-audio recall goes 0.00 → 0.85.

**Scheduled (timer-based) trigger fires.**  Position estimates arrive once
per chunk and ~processing-latency late, so waiting to OBSERVE a boundary
crossing fires ~100ms late on average plus the processing delay.  When a
crossing is predicted within `schedule_horizon_sec`, the scheduler arms a
`threading.Timer` at the exact predicted wall moment (playback rate is 1.0);
each newer estimate re-arms it, and a confidence drop cancels it.  The
offline benchmark can't use wall timers (it replays faster than real time),
so `wall_timers=False` releases pending fires at their wall deadline in
file-time — including any estimate lag, so the benchmark cannot flatter
itself (regression-tested).

**Cold start.**  The HMM starts uniform; nothing trusts position until the
initial lock (consistency + cost margin) succeeds, at which point
`set_prior_from_coarse()` seeds the HMM at the locked slide and the catch-up
trigger shows the current slide immediately.

## Test coverage

| File | What it covers |
|---|---|
| `test_dtw.py` | cosine distance, subsequence DTW, similarity search bounds, `align()` keys, `rigid_align` (exact subsequence, empty window) |
| `test_hmm.py` | transition matrix (row sums, left-to-right, absorbing last state), `update()` keys, state probs sum to 1, convergence, drift, reset, trigger confidence near boundary |
| `test_io.py` | manifest parsing (stops inferred, finalize, empty raises), audio resampling and stereo→mono |
| `test_transform.py` | global mean, L2 norm after contrastive, single vector, zero-out identical rows |
| `test_windows.py` | window count, timestamps monotone, short audio → empty, pool range, pool empty range |
| `test_trigger.py` | fire gating/ordering/cooldown, skip pointer, ProPresenter index mapping via fake controller, boundary selection (catch-up/skips), scheduled fires (virtual + wall timers, re-arm, cancel-on-low-confidence, lag honesty) |
| `test_pp_live.py` | live ProPresenter integration (auto-skips when unreachable): go_to_slide round-trip, TriggerScheduler→controller delivery |
