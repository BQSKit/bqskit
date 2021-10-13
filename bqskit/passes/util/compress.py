"""This module implements the CompressPass class."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class CompressPass(BasePass):
    """
    The CompressPass class.

    The CompressPass compresses the circuit's cycles which might make future
    passes slightly more efficient without actually changing the circuit.
    """

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        circuit.compress()
