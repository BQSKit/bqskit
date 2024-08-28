"""Checks the attached runtime's process management and ability to cleanup."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime


# @pytest.mark.parametrize('num_workers', [1, -1])
# def test_startup_shutdown_transparently(num_workers: int) -> None:
#     in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
#     compiler = Compiler(num_workers=num_workers)
#     assert compiler.p is not None
#     compiler.__del__()
#     if sys.platform == 'win32':
#         time.sleep(1)
#     out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
#     assert in_num_childs == out_num_childs


@pytest.mark.parametrize('num_workers', [1, -1])
def test_cleanup_close(num_workers: int) -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    compiler = Compiler(num_workers=num_workers)
    compiler.close()
    if sys.platform == 'win32':
        time.sleep(1)
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs


@pytest.mark.parametrize('num_workers', [1, -1])
def test_cleanup_with_clause(num_workers: int) -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    with Compiler(num_workers=num_workers) as _:
        pass
    if sys.platform == 'win32':
        time.sleep(1)
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs


@pytest.mark.parametrize('num_workers', [1, 2, 4])
def test_create_workers(num_workers: int) -> None:
    compiler = Compiler(num_workers=num_workers)
    assert compiler.p is not None
    expected = [num_workers, num_workers + 1]  # Some OS create a spawn server
    assert len(psutil.Process(compiler.p.pid).children()) in expected
    compiler.close()


def test_two_thread_per_worker() -> None:
    if sys.platform == 'win32':
        pytest.skip('Not sure how to count threads on Windows.')

    if sys.platform == 'darwin':
        pytest.skip('MacOS requires permissions to count threads.')

    compiler = Compiler(num_workers=1)
    assert compiler.p is not None
    assert len(psutil.Process(compiler.p.pid).children()) in [1, 2]
    assert psutil.Process(compiler.p.pid).children()[0].num_threads() == 2
    compiler.close()


def test_double_close() -> None:
    compiler = Compiler(num_workers=1)
    assert compiler.p is not None
    compiler.close()
    assert compiler.p is None
    compiler.close()
    assert compiler.p is None


def test_interrupt_handling() -> None:
    if sys.platform == 'win32':
        sig = signal.SIGTERM
    else:
        sig = signal.SIGINT

    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    p = subprocess.Popen([
        'python', '-c',
        """
        import time
        from bqskit.compiler import Compiler
        compiler = Compiler(num_workers=1)
        time.sleep(10)
        """,
    ])
    p.send_signal(sig)

    if sys.platform != 'win32':
        out_num_childs = len(
            psutil.Process(
                os.getpid(),
            ).children(recursive=True),
        )
        assert in_num_childs + 1 == out_num_childs

    p.wait()
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs


class TestErrorPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        raise RuntimeError('Boo!')


def raise_error() -> None:
    raise RuntimeError('Boo!')


class TestNestedErrorPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        await get_runtime().submit(raise_error)


class TestWorkerExitPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        exit()


def test_errors_shutdown_system() -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    try:
        compiler = Compiler(num_workers=1)
        compiler.compile(Circuit(2), [TestErrorPass()])
    except RuntimeError:
        if sys.platform == 'win32':
            time.sleep(1)
        num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
        assert in_num_childs == num_childs
    else:
        assert False, 'No error caught.'


def test_errors_shutdown_system_nested() -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    try:
        compiler = Compiler(num_workers=1)
        compiler.compile(Circuit(2), [TestNestedErrorPass()])
    except RuntimeError:
        if sys.platform == 'win32':
            time.sleep(1)
        num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
        assert in_num_childs == num_childs
    else:
        assert False, 'No error caught.'


def test_worker_fail_shutdown_system() -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    try:
        compiler = Compiler(num_workers=2)
        compiler.compile(Circuit(2), [TestWorkerExitPass()])
    except RuntimeError:
        if sys.platform == 'win32':
            time.sleep(1)
        num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
        assert in_num_childs == num_childs
    else:
        assert False, 'No error caught.'
