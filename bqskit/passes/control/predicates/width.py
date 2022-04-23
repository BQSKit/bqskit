"""This module implements the WidthPredicate class."""
from __future__ import annotations

from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate
from bqskit.utils.typing import is_integer


class WidthPredicate(PassPredicate):
    """
    The WidthPredicate class.

    The WidthPredicate class returns True if the circuit's width is less than a
    specified number.
    """

    def __init__(self, width: int) -> None:
        """
        Construct a WidthPredicate.

        Args:
            width (int): Return true if the circuit's width is less than this.
        """

        if not is_integer(width):
            raise TypeError(f'Expected int, got {type(width)}')

        self.width = width

    def get_truth_value(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return circuit.num_qudits < self.width
