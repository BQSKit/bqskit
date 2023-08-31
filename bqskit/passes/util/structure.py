"""This module implements the StructureAnalysisPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
from bqskit.ir.structure import CircuitStructure


class StructureAnalysisPass(BasePass):
    """
    A pass that catalogs circuit structures used in a partitioned circuit.

    Note: For each partitioned CircuitGate in the Circuit, the recursive
    `unfold_all` method is called to ensure that structure is defined at
    the gate level. For structures to be considered the same, they must
    consist of the same gates, in the same locations.
    """

    def __init__(self) -> None:
        """Construct a StructurePass."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if 'structures' not in data:
            data['structures'] = []

        structures_seen: list[CircuitStructure] = []

        for block in circuit:
            if not isinstance(block.gate, CircuitGate):
                continue
            subcirc = Circuit.from_operation(block)
            # Structure depends on the gate level, call unfold_all
            subcirc.unfold_all()
            structure = CircuitStructure(subcirc)
            structures_seen.append(structure)

        data['structures'].extend(structures_seen)
