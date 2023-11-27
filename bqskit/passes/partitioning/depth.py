"""This module defines the ClusteringPartitioner pass."""
from __future__ import annotations

import logging

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class DepthPartitioner(BasePass):
    """
    The DepthPartitioner Pass.

    This pass forms partitions of depth d, with full width
    """

    def __init__(self, depth: int = 100) -> None:
        """
        Construct a ClusteringPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

            num_points (int): Total number of points to place and clusters
                to form. (Default: 8)

        Raises:
            ValueError: If `block_size` is less than 2.

            ValueError: if `num_points` is nonpositive.
        """
        self.depth = depth

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Split depth by cycles
        circuit.compress()

        if self.depth > circuit.depth:
            _logger.warning(
                'Configured depth is greater than circuit depth; '
                'blocking entire circuit.',
            )
            circuit.fold({
                qudit_index: (0, circuit.num_cycles - 1)
                for qudit_index in range(circuit.num_qudits)
            })
            return

        i = 0
        qubits = range(circuit.num_qudits)
        regions = []
        while i < (circuit.depth - 1):
            right = min(i + self.depth, circuit.depth - 1)
            region = {q: (i, right) for q in qubits}
            regions.append(region)
            _logger.debug(f"Compressed Cycles {i} to {right} into a block")
            i = right + 1


        _logger.debug(f"There are {len(regions)} total blocks!")
        for region in reversed(regions):
            circuit.fold(region)

        num_blocks = circuit.num_operations
        _logger.debug(f"There inded are {num_blocks} total blocks!")