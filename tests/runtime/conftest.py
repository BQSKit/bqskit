import signal
import pytest
import subprocess
from typing import Any, Iterator
from bqskit.compiler import Compiler

@pytest.fixture
def detached_compiler() -> Iterator[Compiler]:
    manager = subprocess.Popen(["bqskit-manager", '-n1'])
    server = subprocess.Popen(["bqskit-server", 'localhost'])
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

@pytest.fixture(params=["attached", "detached"])
def server_compiler(request: Any) -> Iterator[Compiler]:
    if request.param[0] == "detached":
        manager = subprocess.Popen(["bqskit-manager", '-n1'])
        server = subprocess.Popen(["bqskit-server", 'localhost'])
        compiler = Compiler('localhost')
    else:
        compiler = Compiler()

    yield compiler

    if request.param[0] == "detached":
        compiler.close()
        server.send_signal(signal.SIGINT)
        server.wait()
        manager.wait()
        manager.kill()
        server.kill()
    else:
        compiler.close()
