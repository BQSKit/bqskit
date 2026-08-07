"""This module contains the logging configuration and methods for BQSKit."""
from __future__ import annotations

import logging
from sys import stdout as _stdout


_logging_initialized = False


def enable_logging(verbose: bool = False) -> None:
    """
    Enable logging for BQSKit.

    Args:
        verbose (bool): If set to True, will print more verbose messages.
            Defaults to False.
    """
    global _logging_initialized
    if not _logging_initialized:
        _logger = logging.getLogger('bqskit')
        _handler = logging.StreamHandler(_stdout)
        _handler.setLevel(0)
        _fmt_header = '%(asctime)s.%(msecs)03d - %(levelname)-8s |'
        _fmt_message = ' %(name)s: %(message)s'
        _fmt = _fmt_header + _fmt_message
        _formatter = logging.Formatter(_fmt, '%H:%M:%S')
        _handler.setFormatter(_formatter)
        _logger.addHandler(_handler)
        _logging_initialized = True

    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger('bqskit').setLevel(level)


def disable_logging() -> None:
    """Disable logging for BQSKit."""
    logging.getLogger('bqskit').setLevel(logging.CRITICAL)
