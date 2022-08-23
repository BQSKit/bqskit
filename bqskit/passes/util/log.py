"""This module implements the LogPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class LogPass(BasePass):
    """A pass that logs a message to the BQSKit log."""

    def __init__(self, msg: str, level: int = logging.INFO) -> None:
        """
        Construct a LogPass.

        Args:
            msg (str): The message to log.

            level (int): The logging verbosity level.
        """
        self.msg = msg
        self.level = level

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.log(self.level, self.msg)
