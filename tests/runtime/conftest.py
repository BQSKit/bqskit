from __future__ import annotations

import signal
import subprocess
import sys
from typing import Any
from typing import Iterator

import pytest

from bqskit.compiler import Compiler

params = ['attached', 'detached'] if sys.platform != 'win32' else ['attached']


@pytest.fixture(params=params)
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
