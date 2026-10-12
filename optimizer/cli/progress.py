"""Display-only live progress line for long terminal calls.

The line goes to a terminal stream only. It records nothing, writes no files and
adds nothing to command history. A background timer keeps the elapsed time
moving even while one long evaluation runs, so a working call never looks frozen.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Any, TextIO

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_WIDTH = 20
_CLEAR = "\r\x1b[2K"


def _number(value: Any) -> str:
    try:
        return format(float(value), ".4g")
    except (TypeError, ValueError):
        return str(value)


class _LiveLine:
    """One redrawn status line for the call made inside its `with` block."""

    def __init__(self, label: str, stream: TextIO | None = None, *,
                 delay: float = 0.5, interval: float = 0.1) -> None:
        self.label = str(label).rsplit(".", 1)[-1]
        self.stream = stream if stream is not None else sys.stderr
        self.delay = float(delay)
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._iteration: dict[str, Any] | None = None
        self._item: dict[str, Any] | None = None
        self._drawn = False
        self._start = 0.0

    def __enter__(self) -> "_LiveLine":
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="optimizer-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._drawn:
            self._write(_CLEAR)
        return False

    def update(self, event: dict[str, Any]) -> None:
        """Receive one library progress event; called from the computing thread."""
        with self._lock:
            if event.get("kind") == "item":
                self._item = dict(event)
            else:
                self._iteration = dict(event)

    def _run(self) -> None:
        frame = 0
        while not self._stop.wait(self.interval):
            elapsed = time.perf_counter() - self._start
            if elapsed < self.delay:
                continue
            with self._lock:
                text = self.text(elapsed, frame)
            self._write(_CLEAR + text)
            self._drawn = True
            frame += 1

    def text(self, elapsed: float, frame: int = 0) -> str:
        """Compose the status line from the latest events."""
        item, iteration = self._item, self._iteration
        primary = item or iteration
        parts = [self.label]
        index = total = None
        if primary is not None:
            index = primary.get("index") if item is not None else primary.get("iteration")
            total = primary.get("total")
        unit = f" {item.get('unit')}s" if item is not None and item.get("unit") else ""
        if total and index is not None and int(total) > 0:
            filled = min(_BAR_WIDTH, int(round(_BAR_WIDTH * min(1.0, int(index) / int(total)))))
            parts.append("█" * filled + "░" * (_BAR_WIDTH - filled))
            parts.append(f"{index}/{total}{unit}")
        else:
            parts.append(_FRAMES[frame % len(_FRAMES)])
            if index is not None:
                parts.append(f"{(item or {}).get('unit') or 'iter'} {index}")
        if item is not None and "qualified" in item:
            parts.append(f"qualified {item['qualified']}")
        if item is not None and iteration is not None:
            parts.append(f"iter {iteration.get('iteration')}")
        if iteration is not None and iteration.get("value") is not None:
            parts.append(f"{iteration.get('metric') or 'value'}={_number(iteration['value'])}")
        parts.append(f"{elapsed:.1f}s")
        if total and index and 0 < int(index) < int(total):
            parts.append(f"~{elapsed / int(index) * (int(total) - int(index)):.1f}s left")
        width = shutil.get_terminal_size((80, 20)).columns
        return "  ".join(parts)[: max(10, width - 1)]

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:  # noqa: BLE001 - a closed terminal must not break the call.
            pass


def _live_line_factory(stream: TextIO | None = None):
    """Return a per-call progress factory for a terminal stream, or None."""
    stream = stream if stream is not None else sys.stderr
    try:
        interactive = bool(stream.isatty())
    except Exception:  # noqa: BLE001
        interactive = False
    if not interactive:
        return None
    return lambda label: _LiveLine(label, stream)
