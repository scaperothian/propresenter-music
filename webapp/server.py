"""ppsync live monitor — tiny stdlib web server.

Tails the JSON-lines telemetry log written by ``ppsync-align --log`` and
streams each frame to the browser over Server-Sent Events.  No third-party
dependencies; works alongside the aligner without touching it.

Usage:
    # terminal 1 — alignment with telemetry
    .venv/bin/ppsync-align data/studio_cache_sliding.npz --mic --dry-run \
        --log /tmp/ppsync.jsonl

    # terminal 2 — monitor UI
    python webapp/server.py --log /tmp/ppsync.jsonl --port 8765
    # then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent

LOG_PATH = Path("/tmp/ppsync.jsonl")  # overridden by --log
REPLAY = False


class Handler(BaseHTTPRequestHandler):
    """GET / → index.html;  GET /events → SSE stream of telemetry lines."""

    def do_GET(self):  # noqa: N802  (stdlib naming)
        if self.path in ("/", "/index.html"):
            body = (WEBAPP_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self._stream_events()
        else:
            self.send_error(404)

    def log_message(self, *args):  # silence per-request stderr noise
        pass

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
    global LOG_PATH, REPLAY  # noqa: PLW0603 — tiny single-purpose server

    p = argparse.ArgumentParser(description="ppsync live monitor web UI.")
    p.add_argument("--log", default=str(LOG_PATH),
                   help="Telemetry JSONL written by ppsync-align --log.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--replay", action="store_true",
                   help="Stream the whole existing log first (default: tail only).")
    args = p.parse_args()

    LOG_PATH = Path(args.log)
    REPLAY = args.replay

    server = ThreadingHTTPServer(("", args.port), Handler)
    print(f"ppsync monitor: http://localhost:{args.port}  (tailing {LOG_PATH})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
