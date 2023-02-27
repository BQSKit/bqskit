"""This module implements the CompilationStatus enum."""
from __future__ import annotations

from enum import Enum


class MyInt(int):
    __reduce_ex__ = int.__reduce_ex__


class CompilationStatus(MyInt, Enum):  # type: ignore
    """The status of a CompilationTask."""
    UNKNOWN = 0
    WAITING = 1
    STARTED = 2
    DONE = 3
    ERROR = 4
