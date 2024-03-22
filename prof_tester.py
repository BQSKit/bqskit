"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

from bqskit import compile, Circuit
from bqskit.compiler import Compiler

# Construct the unitary as an NumPy array
circ = Circuit.from_file("tfxy_5.qasm")

print(circ.gate_counts)
# The compile function will perform synthesis

compiler = Compiler(num_workers=4, run_profiler=True)

synthesized_circuit = compile(circ, max_synthesis_size=3, optimization_level=3)
print(synthesized_circuit.gate_counts)
