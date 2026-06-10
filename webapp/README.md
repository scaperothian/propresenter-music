# ppsync live monitor

Browser dashboard for live alignment runs: last triggered slide in a big box,
plus sparklines for DTW / MERT / HMM confidence, song position, and per-chunk
processing latency.  Stdlib only — no extra dependencies.

## Run

```bash
# terminal 1 — alignment with telemetry logging
.venv/bin/ppsync-align data/studio_cache_sliding.npz --mic --dry-run \
    --log /tmp/ppsync.jsonl

# terminal 2 — monitor
.venv/bin/python webapp/server.py --log /tmp/ppsync.jsonl
```

Open <http://localhost:8765>.

Flags: `--port` (default 8765), `--replay` (stream the whole existing log on
connect instead of tailing from the end — useful to review a finished run).

## How it works

`ppsync-align --log` writes one JSON line per audio chunk (line-buffered).
`server.py` tails that file and forwards each line to the browser over
Server-Sent Events; `index.html` renders it with vanilla JS canvas sparklines.
The aligner itself is untouched — kill or restart either side independently.

Status pill: `buffering` (filling the 6s warm-up window), `silence` (mic open
but level below `SILENCE_RMS_DBFS`), `aligned` (tracking), `stale` (no frames
for >3s — aligner stopped or fell over).
