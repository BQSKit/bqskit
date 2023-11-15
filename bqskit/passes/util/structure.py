"""This module implements the StructureAnalysisPass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
from bqskit.ir.structure import CircuitStructure

_logger = logging.getLogger(__name__)


class StructureAnalysisPass(BasePass):
    """
    A pass that catalogs circuit structures used in a partitioned circuit.

    If a circuit contains hierarchically defined gates, these hierarchies
    are discarded to check gate level equivalence between structures.

    Example:
        The left shows a hierarchically defined gate sequence. The
        right shows the same gate sequence but it is specified at the
        gate level. These two structures are considered equivalent.
           ----------    ----------
        --|-cx-rz-cx-|--|-cx-rz-cx-|--    --cx-rz-cx-cx-rz-cx--
          | |     |  |  | |     |  |   ==   |     |  |     |
        --|-cx-rz-cx-|--|-cx-rz-cx-|--    --cx-rz-cx-cx-rz-cx--
           ----------    ----------
    """

    def __init__(self) -> None:
        """Construct a StructurePass."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        structures_seen: dict[CircuitStructure, int] = {}

        for block in circuit:
            if isinstance(block.gate, CircuitGate):
                subcirc = block.gate._circuit
                # Structure depends on the gate level, call unfold_all
                subcirc.unfold_all()
                structure = CircuitStructure(subcirc)
                if structure not in structures_seen:
                    structures_seen[structure] = 1
                else:
                    structures_seen[structure] += 1

        if 'structures' in data:
            _logger.warning('Overriding `structures` field in PassData.')
        data['structures'] = structures_seen
