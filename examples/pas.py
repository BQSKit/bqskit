"""This script is synthesizes a unitary with PAS."""
from __future__ import annotations

from bqskit import enable_logging
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes.synthesis.pas import PermutationAwareSynthesisPass
enable_logging()

circuit = Circuit.from_file('cxx.qasm')

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler() as compiler:
    compiled_circuit = compiler.compile(
        circuit,
        PermutationAwareSynthesisPass(),
    )
    print(compiled_circuit.gate_counts)
