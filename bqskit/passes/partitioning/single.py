"""This module implements the GroupSingleQuditGatePass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import MeasurementPlaceholder
from bqskit.ir.gates import Reset
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.region import CircuitRegion


class GroupSingleQuditGatePass(BasePass):
    """
    The GroupSingleQuditGatePass Pass.

    This pass groups together consecutive single-qudit gates.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Go through each qudit individually
        for q in range(circuit.num_qudits):

            single_qubit_regions = []
            region_start = None

            for c in range(circuit.num_cycles):
                if circuit.is_point_idle((c, q)):
                    continue

                op = circuit[c, q]
                if (
                    op.num_qudits == 1
                    and not isinstance(op.gate, (BarrierPlaceholder, MeasurementPlaceholder, Reset))
                ):
                    if region_start is None:
                        region_start = c
                else:
                    if region_start is not None:
                        region = CircuitRegion({q: (region_start, c - 1)})
                        single_qubit_regions.append(region)
                        region_start = None

            if region_start is not None:
                region = CircuitRegion(
                    {q: (region_start, circuit.num_cycles - 1)},
                )
                single_qubit_regions.append(region)
                region_start = None

            for region in reversed(single_qubit_regions):
                circuit.fold(region)
