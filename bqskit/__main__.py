from __future__ import annotations

import logging

from scipy.stats import unitary_group

from bqskit import Circuit
from bqskit import CompilationTask
from bqskit import Compiler

logger = logging.getLogger('bqskit')
logger.setLevel(logging.INFO)

# Simple Test
compiler = Compiler()
task1 = CompilationTask.synthesis(unitary_group.rvs(8), method='qfast')
compiler.submit(task1)
print(compiler.status(task1))
print(compiler.result(task1))

task2 = CompilationTask(Circuit(2), [])
print(compiler.compile(task2))

# Compiler().compile(task.qfast_synthesis(utry)).circuit.to_file("qasm")
