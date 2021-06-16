"""This module defines the ClusteringPartitioner pass."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class ClusteringPartitioner(BasePass):
    """
    The ClusteringPartitioner Pass.

    This pass forms partitions in the circuit by placing points in the circuit
    clustering the gates around them.
    """

    def __init__(self, block_size: int = 3, num_points: int = 8) -> None:
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

        if not is_integer(block_size):
            raise TypeError(
                f'Expected integer for block_size, got {type(block_size)}.',
            )

        if block_size < 2:
            raise ValueError(
                f'Expected block_size to be greater than 2, got {block_size}.',
            )

        if not is_integer(num_points):
            raise TypeError(
                f'Expected integer for num_points, got {type(num_points)}.',
            )

        if num_points < 1:
            raise ValueError(
                f'Expected num_points to positive, got {num_points}.',
            )

        self.block_size = block_size
        self.num_points = num_points

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.
        """

        if self.block_size > circuit.get_size():
            _logger.warning(
                'Configured block size is greater than circuit size; '
                'blocking entire circuit.',
            )
            circuit.fold({
                qudit_index: (0, circuit.get_num_cycles())
                for qudit_index in range(circuit.get_size())
            })
            return

        for i in range(self.num_points):
            # Randomly select a point
            # TODO: Tile and pick points better
            cycle = 0
            qudit = 0
            while True:
                cycle = np.random.randint(circuit.get_num_cycles())
                qudit = np.random.randint(circuit.get_size())
                if not circuit.is_point_idle((cycle, qudit)):
                    break

            region = circuit.surround((cycle, qudit), self.block_size)
            circuit.fold(region)
