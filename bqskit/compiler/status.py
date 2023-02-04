"""This module implements the CompilationStatus enum."""
from enum import Enum

class MyInt(int):
    __reduce_ex__ = int.__reduce_ex__

class CompilationStatus(MyInt, Enum):
    """The status of a CompilationTask."""
    UNKNOWN = 0
    WAITING = 1
    STARTED = 2
    DONE = 3
    ERROR = 4
