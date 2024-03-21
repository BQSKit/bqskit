"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

from bqskit import compile, Circuit
from bqskit.compiler import Compiler

# Construct the unitary as an NumPy array
circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFXY_5_timesteps/TFXY_5_0.qasm")

print(circ.gate_counts)
# The compile function will perform synthesis

compiler = Compiler(num_workers=4)

synthesized_circuit = compile(circ, max_synthesis_size=3)
print(synthesized_circuit.gate_counts)
