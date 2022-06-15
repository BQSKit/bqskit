"""This module implements the IterativeScanningGateRemovalPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.passes.alias import PassAlias
from bqskit.passes.control import ChangePredicate
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.control import IfThenElsePass
from bqskit.passes.control import WhileLoopPass
from bqskit.passes.control import WidthPredicate
from bqskit.passes.partitioning import ScanPartitioner
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.util import UnfoldPass
from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)


class IterativeScanningGateRemovalPass(PassAlias):
    """
    The IterativeScanningGateRemovalPass class.

    Starting from one side of the circuit, attempt to remove gates one-by-one.
    Repeat this until the circuit no longer changes. This will partition the
    circuit first if the circuit width is too large.
    """

    def __init__(
        self,
        width_to_partition: int = 5,
        block_size: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Construct a IterativeScanningGateRemovalPass.

        Args:
            width_to_partition (int): Circuits at least as wide as this
                will be partitioned first.

            block_size (int): If the circuit is partitioned, it will
                be partitioned into blocks of this width.

            *args (Any): Arguments passed directly to
                :class:`ScanningGateRemovalPass`.

            *kwargs (Any): Keyword arguments passed directly to
                :class:`ScanningGateRemovalPass`.

        Raises:
            ValueError: If `width_to_partition` or `block_size` is
                nonpositive.

            ValueError: If `width_to_partition` is less than or equal to
                `block_size`.
        """

        if not is_integer(width_to_partition):
            raise TypeError(f'Expected an int, got {type(width_to_partition)}.')

        if not is_integer(block_size):
            raise TypeError(f'Expected an int, got {type(block_size)}.')

        if width_to_partition <= 0:
            raise ValueError(
                f'Expected positive width, got {width_to_partition}.',
            )

        if block_size <= 0:
            raise ValueError(f'Expected positive size, got {block_size}.')

        if width_to_partition <= block_size:
            raise ValueError(
                'Expected width to partition at to be greater'
                ' than block size.',
            )

        scan = ScanningGateRemovalPass(*args, **kwargs)
        self.passes: list[BasePass] = [
            WhileLoopPass(
                ChangePredicate(),
                [
                    IfThenElsePass(
                        WidthPredicate(width_to_partition),
                        scan,
                        [
                            ScanPartitioner(block_size),
                            ForEachBlockPass(scan),
                            UnfoldPass(),
                        ],
                    ),
                ],
            ),
        ]

    def get_passes(self) -> list[BasePass]:
        """Return the passes to be run, see :class:`PassAlias` for more."""
        return self.passes
