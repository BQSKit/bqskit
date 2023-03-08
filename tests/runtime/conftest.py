from __future__ import annotations

import signal
import subprocess
from typing import Any
from typing import Iterator

import pytest

from bqskit.compiler import Compiler


@pytest.fixture
def detached_compiler() -> Iterator[Compiler]:
    manager = subprocess.Popen(['bqskit-manager', '-n1'])
    server = subprocess.Popen(['bqskit-server', 'localhost'])
    compiler = Compiler('localhost')
    yield compiler
    compiler.close()
    server.send_signal(signal.SIGINT)
    server.wait()
    manager.wait()


@pytest.fixture
def attached_compiler() -> Iterator[Compiler]:
    compiler = Compiler()
    yield compiler
    compiler.close()


@pytest.fixture(params=['attached', 'detached'])
def server_compiler(request: Any) -> Iterator[Compiler]:
    if request.param == 'detached':
        manager = subprocess.Popen(['bqskit-manager', '-n2', '-i'])
        server = subprocess.Popen(['bqskit-server', 'localhost', '-i'])
        compiler = Compiler('localhost')
    else:
        compiler = Compiler(num_workers=2, runtime_log_level=1)

    yield compiler

    if request.param == 'detached':
        compiler.close()
        server.send_signal(signal.SIGINT)
        server.wait()
        manager.wait()
        manager.kill()
        server.kill()
    else:
        compiler.close()
