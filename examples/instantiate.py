"""Optimize a 3-qubit circuit to be a toffoli gate."""
from __future__ import annotations

import logging

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.qis.unitary import UnitaryMatrix
# The next two lines start bqskits's logger.
logging.getLogger('bqskit').setLevel(logging.INFO)

# We will optimize towards the toffoli unitary.
toffoli = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])
toffoli = UnitaryMatrix(toffoli)

# Start with the circuit structure
circuit = Circuit(3)
circuit.append_gate(VariableUnitaryGate(2), [1, 2])
circuit.append_gate(VariableUnitaryGate(2), [0, 2])
circuit.append_gate(VariableUnitaryGate(2), [1, 2])
circuit.append_gate(VariableUnitaryGate(2), [0, 2])
circuit.append_gate(VariableUnitaryGate(2), [0, 1])

# Instantiate the circuit template with qfactor
circuit.instantiate(
    toffoli,
    method='qfactor',
    diff_tol_a=1e-12,   # Stopping criteria for distance change
    diff_tol_r=1e-6,    # Relative criteria for distance change
    dist_tol=1e-12,     # Stopping criteria for distance
    max_iters=100000,   # Maximum number of iterations
    min_iters=1000,     # Minimum number of iterations
    slowdown_factor=0,   # Larger numbers slowdown optimization
    # to avoid local minima
)

# Calculate and print final distance
dist = HilbertSchmidtCostGenerator().calc_cost(circuit, toffoli)
print('Final Distance: ', dist)

# If you would like to convert the unitary operations to native gates,
# you should use the KAK decomposition for 2 qubit unitaries, or
# qsearch or qfast for 3+ qubit unitaries.
