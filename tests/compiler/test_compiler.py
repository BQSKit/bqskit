"""This module tests BQSKit's Compiler object."""
from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import pytest

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit


class ErrorPass(BasePass):
    async def run(
        self,
        circuit: Circuit,
        data: dict[str, Any] = {},
    ) -> None:
        raise RuntimeError()


class LogPass(BasePass):
    async def run(
        self,
        circuit: Circuit,
        data: dict[str, Any] = {},
    ) -> None:
        logging.getLogger('bqskit.dummy').debug('Test.')


def test_errors_raised_locally() -> None:
    task = CompilationTask(Circuit(1), [ErrorPass()])
    with pytest.raises(RuntimeError):
        with Compiler() as compiler:
            compiler.compile(task)


def test_log_msg_printed_locally() -> None:
    task = CompilationTask(Circuit(1), [LogPass()])
    logger = logging.getLogger('bqskit')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    with Compiler() as compiler:
        compiler.compile(task)
    assert 'Test.' in handler.stream.getvalue()
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)
