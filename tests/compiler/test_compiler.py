"""This module tests BQSKit's Compiler object."""
from __future__ import annotations

import logging
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit


def test_errors_raised_locally(compiler: Compiler) -> None:
    class ErrorPass(BasePass):
        def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
            raise RuntimeError()

    task = CompilationTask(Circuit(1), [ErrorPass()])

    with pytest.raises(RuntimeError):
        compiler.compile(task)


def test_log_msg_printed_locally(compiler: Compiler) -> None:
    class LogPass(BasePass):
        def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
            logging.getLogger('bqskit.dummy').debug('Test.')

    task = CompilationTask(Circuit(1), [LogPass()])
    logging.getLogger('bqskit').setLevel(logging.DEBUG)
    with patch('sys.stdout', new_callable=StringIO) as mock_out:
        compiler.compile(task)
        print(mock_out.getvalue())
    logging.getLogger('bqskit').setLevel(logging.WARNING)
