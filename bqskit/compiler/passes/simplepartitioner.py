"""This module defines the SimplePartitioner pass."""

from typing import Any
from bqskit.ir.circuit import Circuit
from bqskit.compiler.basepass import BasePass

class SimplePartitioner(BasePass):

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Block gates into CircuitGates."""
