"""Shared CLI logging and dependency-free terminal progress reporting."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import TextIO


_LOGGER_NAME = "rdb_prior"
_HANDLER_MARKER = "_rdb_prior_handler"
_ACTIVE_REPORTER: ProgressReporter | None = None


class _ProgressAwareStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        reporter = _ACTIVE_REPORTER
        redraw = bool(
            reporter is not None
            and reporter.enabled
            and reporter._rendered
            and reporter.stream is self.stream
        )
        if redraw and reporter is not None:
            reporter._clear_line()
        super().emit(record)
        if redraw and reporter is not None:
            reporter._redraw()


def close_logging() -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in tuple(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    normalized = level.upper()
    numeric_level = logging.getLevelName(normalized)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unknown log level: {level!r}")
    logger = logging.getLogger(_LOGGER_NAME)
    close_logging()
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = _ProgressAwareStreamHandler(stream or sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_MARKER, True)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)
    return logger


class ProgressReporter:
    def __init__(
        self,
        *,
        stage: str,
        total: int | None = None,
        logger: logging.Logger | None = None,
        log_every: int = 100,
        enabled: bool | None = None,
        width: int = 28,
        stream: TextIO | None = None,
    ) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if total is not None and total < 0:
            raise ValueError("total must be non-negative")
        if log_every < 0:
            raise ValueError("log_every must be non-negative")
        if width < 10:
            raise ValueError("progress width must be at least 10")
        self.stage = stage
        self.total = total
        self.logger = logger or logging.getLogger(f"{_LOGGER_NAME}.progress")
        self.log_every = log_every
        self.stream = stream or sys.stderr
        self.enabled = (
            bool(self.stream.isatty()) if enabled is None else enabled
        )
        self.width = width
        self.started_at = time.monotonic()
        self.completed = 0
        self._closed = False
        self._rendered = False
        self._last_line = ""
        if self.enabled:
            global _ACTIVE_REPORTER
            _ACTIVE_REPORTER = self

    def update(
        self,
        completed: int,
        total: int,
        item_id: str,
        *,
        detail: str | None = None,
    ) -> None:
        if completed < 0 or total < 0 or completed > total:
            raise ValueError("progress must satisfy 0 <= completed <= total")
        self.completed = completed
        self.total = total
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else None
        if self.enabled:
            self._render(item_id=item_id, rate=rate, eta=eta, detail=detail)
        elif self._is_log_milestone(completed, total):
            suffix = f"; {detail}" if detail else ""
            self.logger.info(
                "%s progress %d/%d (%.1f%%); item=%s; rate=%.2f/s%s",
                self.stage,
                completed,
                total,
                _percentage(completed, total),
                item_id,
                rate,
                suffix,
            )

    def close(self) -> None:
        if self._closed:
            return
        if self.enabled and self._rendered:
            self.stream.write("\n")
            self.stream.flush()
        global _ACTIVE_REPORTER
        if _ACTIVE_REPORTER is self:
            _ACTIVE_REPORTER = None
        self._closed = True

    def _render(
        self,
        *,
        item_id: str,
        rate: float,
        eta: float | None,
        detail: str | None,
    ) -> None:
        total = self.total or 0
        fraction = self.completed / total if total else 0.0
        filled = min(self.width, round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" | {detail}" if detail else ""
        line = (
            f"[{self.stage}] [{bar}] {self.completed}/{total} "
            f"{_percentage(self.completed, total):5.1f}% "
            f"{rate:6.2f}/s ETA {_duration(eta)} | {item_id}{suffix}"
        )
        self._last_line = line
        self.stream.write("\r" + line)
        if self.completed == total:
            self.stream.write("\n")
        self.stream.flush()
        self._rendered = self.completed != total

    def _clear_line(self) -> None:
        self.stream.write("\r" + " " * len(self._last_line) + "\r")
        self.stream.flush()

    def _redraw(self) -> None:
        if self._last_line:
            self.stream.write("\r" + self._last_line)
            self.stream.flush()

    def _is_log_milestone(self, completed: int, total: int) -> bool:
        return completed == total or (
            self.log_every > 0 and completed % self.log_every == 0
        )


def _percentage(completed: int, total: int) -> float:
    return 100.0 * completed / total if total else 100.0


def _duration(value: float | None) -> str:
    if value is None:
        return "--:--"
    seconds = max(0, round(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


__all__ = ["configure_logging", "close_logging", "ProgressReporter"]
