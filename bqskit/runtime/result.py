"""This module implements the RuntimeResult NamedTuple."""
from __future__ import annotations

from typing import Any
from typing import NamedTuple

from bqskit.runtime.address import RuntimeAddress


class RuntimeResult(NamedTuple):
    """The result of a task, ready to be shipped to its destination."""
    return_address: RuntimeAddress
    result: Any
    completed_by: int
