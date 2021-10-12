from __future__ import annotations

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir import Circuit

# Load a circuit from QASM
circuit = Circuit.from_file('tfim.qasm')

# Create a standard CompilationTask to optimize the circuit
task = CompilationTask.optimize(circuit)

# Spawn a compiler and compile the task
with Compiler() as compiler:
    optimized_circuit = compiler.compile(task)
