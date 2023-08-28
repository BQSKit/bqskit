"""This module implements the StructurePass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate


class StructurePass(BasePass):
    """A pass that catalogs circuit structures used in a partitioned circ-
    uit."""

    def __init__(self) -> None:
        """Construct a StructurePass."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if 'structures' not in data:
            data['structures'] = []

        structures_seen = []

        for block in circuit:
            if not isinstance(block.gate, CircuitGate):
                continue
            subcirc = Circuit.from_operation(block)
            subcirc.unfold_all()
            subcirc.set_params([0] * subcirc.num_params)

            structures_seen.append(subcirc)

        data['structures'].extend(structures_seen)
