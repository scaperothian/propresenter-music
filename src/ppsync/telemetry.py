"""JSON-lines telemetry logger.

Each frame logged contains enough information to replay the full pipeline
offline:  MERT embedding quality, DTW position + cost, HMM state, trigger
decisions.  All numeric arrays are serialised as plain lists so the log
stays human-readable with standard JSON tooling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class TelemetryLogger:
    """Append one JSON line per frame to a log file (or stdout)."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._f = None
        if log_path is not None:
            self._f = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(self, frame: dict[str, Any]) -> None:
        line = json.dumps(_serialize(frame), separators=(",", ":"))
        if self._f is not None:
            self._f.write(line + "\n")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _serialize(obj: Any) -> Any:
    """Recursively convert numpy / torch objects to JSON-safe types."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj
