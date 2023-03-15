"""This module implements the CompilationStatus enum."""
from __future__ import annotations

from enum import IntEnum


class CompilationStatus(IntEnum):
    """The status of a CompilationTask."""
    UNKNOWN = 0
    RUNNING = 1
    DONE = 2
