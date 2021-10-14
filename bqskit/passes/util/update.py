"""This module implements the UpdateDataPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class UpdateDataPass(BasePass):
    """
    The UpdateDataPass class.

    The UpdateDataPass adds a key-value pair to data dictionary.
    """

    def __init__(self, key: str, val: Any) -> None:
        """
        Construct a UpdateDataPass.

        Args:
            key (str): The key to add.

            val (Any): The value to associate with the key.
        """

        if not isinstance(key, str):
            raise TypeError('Expected string for key, got %s.' % type(key))

        self.key = key
        self.val = val

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug(f'Injecting {self.key}:{self.val} into the data dict.')
        data[self.key] = self.val
