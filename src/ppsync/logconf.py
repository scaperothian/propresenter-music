"""Logging configuration for ppsync.

Library modules log via ``logging.getLogger(__name__)`` (all children of the
``ppsync`` logger) and never configure handlers themselves.  The CLI entry
points call :func:`configure_logging` once at startup.

Real-time safety
----------------
Records are pushed onto a queue by a ``QueueHandler`` (a near-free, lock-light
enqueue) and formatted/written on a background ``QueueListener`` thread, so the
audio loop never blocks on terminal or file I/O.  Combined with level-gating
(the per-chunk status line logs at DEBUG and is skipped entirely when DEBUG is
off), this keeps the 200ms processing loop free of synchronous output cost.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
from typing import TextIO

# The package-root logger.  Module loggers (ppsync.aligner, ppsync.trigger, …)
# propagate up to it; configuring it once configures them all.
ROOT_LOGGER_NAME = "ppsync"

_listener: logging.handlers.QueueListener | None = None


def _coerce_level(level: int | str) -> int:
    """Accept an int level or a name like 'info' / 'DEBUG'."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> None:
    """Attach a non-blocking queue handler to the ``ppsync`` logger.

    Args:
        level:  threshold for the ``ppsync`` logger (int or name).
        stream: where formatted records are written (default: stdout, so logs
                can be routed to the screen for CLI use).

    Idempotent: a second call tears down the previous listener and handlers
    first, so repeated calls (e.g. in tests) do not stack handlers.
    """
    global _listener

    if stream is None:
        stream = sys.stdout

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(_coerce_level(level))
    root.propagate = False  # don't also bubble to the Python root logger

    # Tear down any previous configuration (idempotent / re-entrant).
    _stop_logging()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    log_queue: queue.Queue = queue.Queue(-1)  # unbounded; enqueue never blocks
    root.addHandler(logging.handlers.QueueHandler(log_queue))

    stream_handler = logging.StreamHandler(stream)
    # Bare message keeps CLI output looking like the prints it replaces;
    # warnings/errors carry their own "WARNING:"/"error" text inline.
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    _listener = logging.handlers.QueueListener(
        log_queue, stream_handler, respect_handler_level=True
    )
    _listener.start()
    atexit.register(_stop_logging)


def _stop_logging() -> None:
    """Stop the background listener, flushing any queued records."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
