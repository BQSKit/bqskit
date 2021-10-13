"""This module implements the NOOPPass class."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class NOOPPass(BasePass):
    """NOOPPass class, does not perform any operation."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        return
