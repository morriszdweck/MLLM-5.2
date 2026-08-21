"""ANSI terminal styling with tty / NO_COLOR auto-detection."""

from __future__ import annotations

import os
import sys

_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
}
_RESET = "\033[0m"


class Term:
    """Paints text with ANSI colors when enabled, else returns it unchanged."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @classmethod
    def detect(cls, no_color: bool = False, stream=None) -> "Term":
        """Create a Term whose colors are on only for interactive terminals."""
        if stream is None:
            stream = sys.stdout
        is_tty = hasattr(stream, "isatty") and stream.isatty()
        enabled = not no_color and is_tty and os.environ.get("NO_COLOR") is None
        return cls(enabled=enabled)

    def paint(self, color: str, text: str) -> str:
        """Wrap *text* in the named color."""
        if not self.enabled:
            return text
        code = _CODES.get(color, "")
        return f"{code}{text}{_RESET}" if code else text
