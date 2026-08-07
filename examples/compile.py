"""
This example demonstrates using the standard BQSKit compile function.

For a much more detailed tutorial, see the bqskit-tutorial repository.
"""
from __future__ import annotations

from bqskit import Circuit
from bqskit import compile

# Load a circuit from QASM
circuit = Circuit.from_file('tfim.qasm')

# Compile it with optimization level 2
out_circuit = compile(circuit, optimization_level=2)

# You can choose to save the output as qasm
out_circuit.save('tfim_out.qasm')
