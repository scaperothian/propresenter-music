# Benchmarks

Regression benchmarks for the ppsync alignment pipeline. Two dimensions:

- **Accuracy** — does the aligner know where it is, and do slides fire on time?
- **Latency** — does each audio chunk process inside the real-time budget?

`REPORT.md` (checked in) is the latest generated result; `results.json` is the
same data machine-readable for diffing / CI gates.

## Tools

| Tool | Scope | Output |
|---|---|---|
| `tools/benchmark.py` | accuracy for ONE song (start-offset re-sync sweep) + latency side-metrics | stdout, `--json-out`, `--trace-out` |
| `tools/latency_benchmark.py` | per-chunk latency for ONE song, with an embed-vs-rest breakdown | stdout, `--json-out` |
| `tools/benchmark_report.py` | runs accuracy + latency across a WHOLE dataset and writes the aggregated report | `benchmarks/REPORT.md` + `results.json` |

## Regenerating the report

```bash
# Whole dataset (builds manifests + caches under data/ as needed, ~45 min cold)
.venv/bin/python tools/benchmark_report.py --dataset ../slide-agent/dataset

# Faster iteration: a subset, and reuse existing caches
.venv/bin/python tools/benchmark_report.py --dataset ../slide-agent/dataset \
    --only cocaine,layla

# Compare matchers / a DTW step penalty
.venv/bin/python tools/benchmark_report.py --dataset ../slide-agent/dataset \
    --matcher dtw --dtw-step-penalty 0.1

# Add re-sync offsets (operator joins mid-song) to the accuracy pooling
.venv/bin/python tools/benchmark_report.py --dataset ../slide-agent/dataset \
    --offsets 0,30,90
```

A song is included only if its folder holds **both** an audio file and a
ProPresenter annotation JSON; audio-less songs are ignored by design.

## Latency for a single song

```bash
.venv/bin/python tools/latency_benchmark.py \
    data/incubus/drive/incubus_drive_cache.npz --file <song>.wav
```

Reports `embed` (isolated MERT forward — the dominant stage), `end-to-end`
(full `process_chunk()`), and `rest` (matcher + HMM + trigger). The headline
is end-to-end **p95 vs the chunk budget** (`CHUNK_SEC`, 200 ms) and **RT-OK%**
(fraction of chunks finishing within budget).

## What the metrics mean

**Accuracy**
- **Recall** — reachable boundaries that fired within the trigger window
  (±`--window-ms`, default 750 ms), pooled over offsets. Boundaries before the
  start offset or inside the warm-up window are unreachable and excluded.
- **Fire MAE / max** — `|fire_time − true_boundary|` over the hits.
- **Track med** — median `|estimated_position − true_position|` after warm-up.
- **Lock-on** — seconds from start until tracking error first drops within 1 s.
- **Spurious** — fires matching no reachable boundary in-window.

**Latency** (per 200 ms chunk)
- **Embed p50** — median isolated MERT forward time.
- **End-to-end p50 / p95 / max** — full per-chunk processing time.
- **RT-OK** — fraction of chunks processed within the budget.

## CI notes

`benchmark_report.py` exits non-zero only on setup failure (no songs found,
nothing produced); per-song errors are caught and the song is skipped so one
bad input can't sink the run. To gate a pipeline, read `results.json` and
assert thresholds (e.g. overall recall, worst-song p95) — the JSON carries
per-song and per-offset detail for that.

The dataset itself is **not** in this repo (it lives in a sibling
`slide-agent/dataset`), and the `data/` cache tree is gitignored, so a CI job
must make the dataset available and will rebuild caches on first run.
