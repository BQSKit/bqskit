from __future__ import annotations

import os
import signal
import subprocess

import psutil
import pytest

from bqskit.compiler import Compiler


# [x] Check children processes
# [ ] Errors in attached shutdown entire system (error in pass)
# [ ] Worker graceful shutdown is handled gracefully (exit in pass)
# [x] Interrupt is handled safely and quickly
# [x] Termination is handled safely and quickly
# [x] All worker processes limit number blas threads to 1


@pytest.mark.parametrize('num_workers', [1, -1])
def test_startup_shutdown_transparently(num_workers: int) -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    compiler = Compiler(num_workers=num_workers)
    assert compiler.p is not None
    compiler.__del__()
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs


@pytest.mark.parametrize('num_workers', [1, -1])
def test_cleanup_close(num_workers: int) -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    compiler = Compiler(num_workers=num_workers)
    compiler.close()
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs


@pytest.mark.parametrize('num_workers', [1, -1])
def test_cleanup_with_clause(num_workers: int) -> None:
    in_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    with Compiler(num_workers=num_workers) as _:
        pass
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs


@pytest.mark.parametrize('num_workers', [1, 2, 4])
def test_create_workers(num_workers: int) -> None:
    compiler = Compiler(num_workers=num_workers)
    assert len(psutil.Process(compiler.p.pid).children()) == num_workers
    compiler.close()


def test_one_thread_per_worker() -> None:
    compiler = Compiler(num_workers=1)
    assert len(psutil.Process(compiler.p.pid).children()) == 1
    assert psutil.Process(compiler.p.pid).children()[0].num_threads() == 1
    compiler.close()


def test_double_close() -> None:
    compiler = Compiler(num_workers=1)
    assert compiler.p is not None
    compiler.close()
    assert compiler.p is None
    compiler.close()
    assert compiler.p is None


def test_interrupt_handling() -> None:
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
    p.send_signal(signal.SIGINT)
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs + 1 == out_num_childs
    p.wait()
    out_num_childs = len(psutil.Process(os.getpid()).children(recursive=True))
    assert in_num_childs == out_num_childs
