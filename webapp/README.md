# ppsync web UI

One stdlib server, two surfaces — no extra dependencies.

- **Live monitor** (`/`): last triggered slide in a big box, plus sparklines
  for DTW / MERT / HMM confidence, song position, and per-chunk latency.
- **Offline analysis** (`/analysis`): three stacked panels on a shared
  true-time axis from benchmark trace files — position trajectory (DTW
  *stalling* shows as flat segments), matcher confidence, and HMM trigger
  confidence — with ground-truth boundaries marked; overlay any number of
  rigid/DTW runs.

## Run

```bash
# live monitor — terminal 1: alignment with telemetry logging
.venv/bin/ppsync-align data/incubus/drive/incubus_drive_cache.npz --mic \
    --dry-run --log /tmp/ppsync_incubus_drive.jsonl
# terminal 2: monitor
.venv/bin/python webapp/server.py --log /tmp/ppsync_incubus_drive.jsonl
# → http://localhost:8765

# offline analysis — produce traces, then point the server at their directory
.venv/bin/python tools/benchmark.py <cache>.npz --file <live>.wav \
    --manifest <manifest>.json --offsets 0 --matcher rigid \
    --trace-out /tmp/ppsync_traces/live_rigid.json
.venv/bin/python tools/benchmark.py <cache>.npz --file <live>.wav \
    --manifest <manifest>.json --offsets 0 --matcher dtw \
    --trace-out /tmp/ppsync_traces/live_dtw.json
.venv/bin/python webapp/server.py --trace-dir /tmp/ppsync_traces
# → http://localhost:8765/analysis  (tick both traces to overlay them)
```

Flags: `--port` (default 8765), `--log` (live monitor source), `--trace-dir`
(offline analysis source), `--replay` (stream the whole existing log on
connect instead of tailing from the end — useful to review a finished run).

## How it works

**Live monitor.** `ppsync-align --log` writes one JSON line per audio chunk
(line-buffered).  `server.py` tails that file and forwards each line to the
browser over Server-Sent Events; `index.html` renders it with vanilla JS
canvas sparklines.  The aligner itself is untouched — kill or restart either
side independently.

Status pill: `buffering` (filling the 6s warm-up window), `silence` (mic open
but level below `SILENCE_RMS_DBFS`), `aligned` (tracking), `stale` (no frames
for >3s — aligner stopped or fell over).

**Offline analysis.** `benchmark.py --trace-out` writes a self-describing
`{meta, frames}` trace (matcher, song, studio slide boundaries, live
ground-truth boundaries, per-frame position/confidence/triggers).  `server.py`
lists the trace dir at `/api/traces` and serves each file at `/api/trace`;
`analysis.html` renders three stacked panels sharing one true-time x-axis:

1. **Position trajectory** — estimated position (`dtw_refined_t`, reference
   time) vs. true time.  A dashed line is the ideal mapping (each slide's live
   time → its reference time); where a matcher's curve goes flat against that
   rising ideal, it is **stalling**.  Trigger fires are dotted on the curve.
2. **Matcher confidence** (`dtw_confidence`) with its 0.55 gate.
3. **HMM trigger confidence** (`hmm_trigger_confidence`) with its 0.60 gate.

Solid white verticals mark every ground-truth slide boundary in all three
panels, so you can read where each confidence sits relative to its gate at a
boundary.  Overlay any number of traces (each a distinct, non-repeating color,
labelled by filename stem) — e.g. rigid vs. DTW, or a DTW step-penalty sweep —
and hover for a per-trace readout (position, m-conf, h-conf, true vs. fired
slide).
