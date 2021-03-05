"""This module defines the SimplePartitioner pass."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class SimplePartitioner(BasePass):

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Block gates into CircuitGates."""
