"""This module implements the LogPass and LogErrorPass classes."""
from __future__ import annotations

import logging

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
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

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.log(self.level, self.msg)


class LogErrorPass(BasePass):
    """A pass that logs the current error to the BQSKit log."""

    def __init__(self, error_threshold: float | None = None) -> None:
        """
        Construct a LogErrorPass.

        Args:
            error_threshold (float | None): Logs a warning if the error
                is above this threshold. If None, never log a warning.
                (Default: None)
        """
        self.threshold = error_threshold

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if 'error' in data:
            error = data['error']
            nonsq_error = 1 - np.sqrt(max(1 - (error * error), 0))
            if self.threshold is not None and nonsq_error > self.threshold:
                _logger.warn(
                    'Upper bound on error is greater than set threshold:'
                    f' {nonsq_error} > {self.threshold}.',
                )
            else:
                _logger.info(f'Upper bound on error is {nonsq_error}.')
