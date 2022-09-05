"""This module implements the LogPass and LogErrorPass classes."""
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

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if 'error' in data:
            error = data['error']
            if self.threshold is not None and error > self.threshold:
                _logger.warn(
                    'Upper bound on error is greater than set threshold:'
                    f' {error} > {self.threshold}.',
                )
            else:
                _logger.info(f'Upper bound on error is {error}.')
