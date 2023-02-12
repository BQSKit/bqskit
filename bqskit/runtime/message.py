"""This module implements the RuntimeMessage enum."""
from __future__ import annotations

from enum import Enum


class MyInt(int):
    __reduce_ex__ = int.__reduce_ex__


class RuntimeMessage(MyInt, Enum):  # type: ignore
    """An message sent between processes in the BQSKit runtime."""
    CONNECT = 0
    DISCONNECT = 1
    STARTED = 2
    SHUTDOWN = 3
    ERROR = 4
    REQUEST = 5
    RESULT = 6
    SUBMIT = 7
    SUBMIT_BATCH = 8
    STATUS = 9
    LOG = 10
    CANCEL = 11
