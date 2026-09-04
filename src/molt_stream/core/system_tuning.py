"""Scoped, opt-in tuning of MOLT only; never mutate other apps or antivirus."""
from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from collections.abc import Iterator

import psutil


@contextmanager
def prioritized_execution(enabled: bool) -> Iterator[bool]:
    """Restore this process's priority on exit, including training exceptions.

    A killed process leaves no suspended apps or persistent host settings.
    Unsupported or denied changes leave training at its original priority.
    """
    process = None
    original = None
    changed = False
    if enabled and sys.platform == "win32":
        try:
            process = psutil.Process()
            original = process.nice()
            process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            changed = True
        except (psutil.Error, OSError):
            warnings.warn("Process priority unavailable; keeping original priority.", RuntimeWarning)
    try:
        yield changed
    finally:
        if changed and process is not None:
            try:
                process.nice(original)
            except (psutil.Error, OSError):
                warnings.warn("Could not restore MOLT process priority; close MOLT to reset it.", RuntimeWarning)
