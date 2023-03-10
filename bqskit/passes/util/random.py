"""This module implements the SetRandomSeedPass class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_integer


_logger = logging.getLogger(__name__)


class SetRandomSeedPass(BasePass):
    """
    The SetRandomSeedPass class.

    The SetRandomSeedPass sets the random seed.
    """

    def __init__(self, seed: int = 0) -> None:
        """
        Construct a SetRandomSeedPass.

        Args:
            seed (int): The value to set the random seed to.
        """

        if not is_integer(seed):
            raise TypeError('Expected integer seed, got %s.' % type(seed))

        self.seed = seed

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug(f'Setting the random seed to {self.seed}.')
        data.seed = self.seed
