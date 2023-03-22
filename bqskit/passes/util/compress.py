"""This module implements the CompressPass class."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit


class CompressPass(BasePass):
    """
    The CompressPass class.

    The CompressPass compresses the circuit's cycles which might make future
    passes slightly more efficient without actually changing the circuit.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        circuit.compress()
