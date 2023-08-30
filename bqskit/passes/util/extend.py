"""This module implements the ExtendBlockSizePass class."""
from __future__ import annotations

import logging
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.qis.graph import CouplingGraph
from bqskit.utils.typing import is_integer


_logger = logging.getLogger(__name__)


class ExtendBlockSizePass(BasePass):
    """Ensure all blocks are at least a given size."""

    def __init__(self, minimum_size: int | None = None) -> None:
        """
        Construct a ExtendBlockSizePass.

        Args:
            minimum_size (int | None): Extend all blocks to at least this
                size. If left as None, the minimum size will be the size of
                the smallest multi-qudit gate in the model.
        """
        if not is_integer(minimum_size) and minimum_size is not None:
            raise TypeError('Expected an integer or None for minimum size.')

        self.minimum_size = minimum_size

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        minimum_size = self.minimum_size
        if minimum_size is None:
            minimum_size = min(
                g.num_qudits for g in data.gate_set
                if g.num_qudits != 1
            )

        if circuit.num_qudits < minimum_size:
            raise RuntimeError('Cannot extend block larger than circuit.')

        cg = self.get_connectivity(circuit, data)

        # Find all small blocks
        small_blocks: list[tuple[int, int]] = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, CircuitGate):
                if op.gate.num_qudits < minimum_size:
                    small_blocks.append((cycle, op.location[0]))
        small_blocks.sort()

        # Select qubits to add
        qudits_to_extend: list[tuple[int, ...]] = []
        for block_point in small_blocks:
            cycle = block_point[0]
            op = circuit[block_point]

            num_to_add = minimum_size - op.gate.num_qudits
            qudits = list(op.location)
            added = []
            for _ in range(num_to_add):
                neighbors = self.get_neighbors(qudits, cg)

                if len(neighbors) == 0:
                    raise RuntimeError('Coupling graph is not fully connected.')

                if any(circuit.is_point_idle((cycle, n)) for n in neighbors):
                    for n in neighbors:
                        if circuit.is_point_idle((cycle, n)):
                            added.append(n)
                            qudits.append(n)
                            break
                else:
                    added.append(neighbors[0])
                    qudits.append(neighbors[0])
            qudits_to_extend.append(tuple(added))

        # Build extended operations
        new_ops: list[Operation] = []
        for block_point, to_add_qudits in zip(small_blocks, qudits_to_extend):
            op = circuit[block_point]
            op_circ: Circuit = op.gate._circuit.copy()  # type: ignore
            for qudit in to_add_qudits:
                op_circ.append_qudit(circuit.radixes[qudit])
            gate = CircuitGate(op_circ, True)
            location = tuple(op.location) + to_add_qudits
            new_op = Operation(gate, location, op.params)
            new_ops.append(new_op)

        # Extend operations
        circuit.batch_replace(small_blocks, new_ops)

    def get_neighbors(self, l: Sequence[int], cg: CouplingGraph) -> list[int]:
        """Return the neighbors of location `l` in `cg`."""
        neighbors = set()
        for q in l:
            neighbors.update(cg.get_neighbors_of(q))
        return list(neighbors.difference(l))
