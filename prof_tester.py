"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

from bqskit import compile, Circuit
from bqskit.compiler import Compiler
import sys

# Construct the unitary as an NumPy array
circ = Circuit.from_file("tfxy_5.qasm")

print(circ.gate_counts)

num_workers = sys.argv[1]
# The compile function will perform synthesis

# Create a controlled workflow. 

# Model: Log to Timeline

compiler = Compiler(num_workers=num_workers, run_profiler=True)

synthesized_circuit = compile(circ, max_synthesis_size=3, optimization_level=3, compiler=compiler)
print(synthesized_circuit.gate_counts)
