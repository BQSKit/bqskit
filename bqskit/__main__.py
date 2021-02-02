from __future__ import annotations

import logging

from bqskit import Circuit
from bqskit import CompilationTask
from bqskit import Compiler

logger = logging.getLogger('bqskit')
logger.setLevel(logging.INFO)
# Simple Test
compiler = Compiler()
task1 = CompilationTask(Circuit(2), 'QASM')
task2 = CompilationTask(Circuit(2), 'QASM')
compiler.submit(task1)
compiler.submit(task2)
print(compiler.status(task1))
print(compiler.status(task2))
print(compiler.result(task1))
print(compiler.result(task2))
del compiler

# Example workflow
# import bqskit

# task = bqskit.CompilationTask(...)
# bqskit.submit( task )
# bqskit.status( task ) Maybe default bqskit backend
# bqskit.result( task ) as static global

# backend = bqskit.Backend() # default local machine... spawns thread
# backend = bqskit.Backend("ip address") # connection other machine in future

# backend.submit( task ) <---
# task.submit( backend )

# ...

# if backend.status(task) == DONE:
# result = backend.result(task)
# do stuff
