"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

from bqskit import compile

# Construct the unitary as an NumPy array
toffoli = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
]

# The compile function will perform synthesis
synthesized_circuit = compile(toffoli, max_synthesis_size=3)
print(synthesized_circuit.gate_counts)
