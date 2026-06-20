"""ppsync web UI — tiny stdlib web server.  Two surfaces, no third-party deps:

1. Live monitor (``/``): tails the JSON-lines telemetry log written by
   ``ppsync-align --log`` and streams each frame over Server-Sent Events.

2. Offline analysis (``/analysis``): loads the self-describing trace files
   written by ``tools/benchmark.py --trace-out`` and plots the alignment
   trajectory (position estimate vs. true time) so the DTW stalling effect is
   visible and rigid/DTW runs can be overlaid.

Usage:
    # live monitor (terminal 2, alongside `ppsync-align --log`)
    python webapp/server.py --log /tmp/ppsync_incubus_drive.jsonl
    # → http://localhost:8765

    # offline analysis (after running benchmark.py --trace-out)
    python webapp/server.py --trace-dir /tmp/ppsync_traces
    # → http://localhost:8765/analysis
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEBAPP_DIR = Path(__file__).resolve().parent

LOG_PATH = Path("/tmp/ppsync.jsonl")  # overridden by --log
REPLAY = False
TRACE_DIR = Path("/tmp/ppsync_traces")  # overridden by --trace-dir


class Handler(BaseHTTPRequestHandler):
    """Live monitor (/ + /events) and offline analysis (/analysis + /api/*)."""

    def do_GET(self):  # noqa: N802  (stdlib naming)
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_html("index.html")
        elif path in ("/analysis", "/analysis.html"):
            self._send_html("analysis.html")
        elif path == "/events":
            self._stream_events()
        elif path == "/api/traces":
            self._list_traces()
        elif path == "/api/trace":
            self._send_trace()
        else:
            self.send_error(404)

    def log_message(self, *args):  # silence per-request stderr noise
        pass

    def _send_html(self, filename: str) -> None:
        body = (WEBAPP_DIR / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _list_traces(self) -> None:
        """List trace files in TRACE_DIR with their meta (matcher/song) header."""
        traces = []
        for p in sorted(TRACE_DIR.glob("*.json")):
            entry = {"name": p.name}
            try:
                obj = json.loads(p.read_text())
                meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
                entry["meta"] = meta
                entry["frames"] = len(obj.get("frames", [])) if isinstance(obj, dict) else 0
            except (json.JSONDecodeError, OSError):
                entry["meta"] = None  # not a ppsync trace — list but mark unusable
            traces.append(entry)
        self._send_json({"trace_dir": str(TRACE_DIR), "traces": traces})

    def _send_trace(self) -> None:
        """Serve one trace file by ?name=, restricted to TRACE_DIR."""
        qs = parse_qs(urlparse(self.path).query)
        name = (qs.get("name") or [""])[0]
        # Resolve and confine to TRACE_DIR (no path traversal).
        target = (TRACE_DIR / name).resolve()
        if not name or TRACE_DIR.resolve() not in target.parents or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Wait for the log file to appear (aligner may not be running yet).
        while not LOG_PATH.exists():
            try:
                self.wfile.write(b": waiting for log file\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(1.0)

        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                if not REPLAY:
                    f.seek(0, 2)  # tail from the end
                last_beat = time.monotonic()
                while True:
                    line = f.readline()
                    if line:
                        if line.endswith("\n"):  # skip partially-written lines
                            self.wfile.write(b"data: " + line.strip().encode() + b"\n\n")
                            self.wfile.flush()
                            last_beat = time.monotonic()
                    else:
                        time.sleep(0.05)
                        if time.monotonic() - last_beat > 10.0:
                            self.wfile.write(b": heartbeat\n\n")  # detects dead clients
                            self.wfile.flush()
                            last_beat = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    global LOG_PATH, REPLAY, TRACE_DIR  # noqa: PLW0603 — tiny single-purpose server

    p = argparse.ArgumentParser(description="ppsync web UI (live monitor + offline analysis).")
    p.add_argument("--log", default=str(LOG_PATH),
                   help="Telemetry JSONL written by ppsync-align --log (live monitor).")
    p.add_argument("--trace-dir", default=str(TRACE_DIR),
                   help="Directory of benchmark --trace-out files (offline analysis).")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--replay", action="store_true",
                   help="Stream the whole existing log first (default: tail only).")
    args = p.parse_args()

    LOG_PATH = Path(args.log)
    REPLAY = args.replay
    TRACE_DIR = Path(args.trace_dir)

    server = ThreadingHTTPServer(("", args.port), Handler)
    print(f"ppsync web UI: http://localhost:{args.port}")
    print(f"  live monitor      /          (tailing {LOG_PATH})")
    print(f"  offline analysis  /analysis  (traces in {TRACE_DIR})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
