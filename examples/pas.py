"""This script is synthesizes a unitary with PAS."""
from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.passes.synthesis.pas import PermutationAwareSynthesisPass
from bqskit.passes.synthesis import LEAPSynthesisPass

from bqskit import enable_logging
enable_logging()

circuit = Circuit.from_file('cxx.qasm')

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler() as compiler:
    compiled_circuit = compiler.compile(circuit, PermutationAwareSynthesisPass())
    print(compiled_circuit.gate_counts)
