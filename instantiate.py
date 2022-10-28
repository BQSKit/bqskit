# %%
"""
Numerical Instantiation is the foundation of many of BQSKit's algorithms.

This example demonstrates building a circuit template that can implement the
toffoli gate and then instantiating it to be the gate.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import unitary_group

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix
jax.config.update('jax_enable_x64', True)


matlib = jnp

# We will optimize towards the toffoli unitary.
toffoli = matlib.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])


# Start with the circuit structure
circuit = Circuit(3)
circuit.append_gate(VariableUnitaryGate(2), [1, 2])
circuit.append_gate(VariableUnitaryGate(2), [0, 2])
circuit.append_gate(VariableUnitaryGate(2), [1, 2])
circuit.append_gate(VariableUnitaryGate(2), [0, 2])
circuit.append_gate(VariableUnitaryGate(2), [0, 1])


# target = UnitaryMatrix(unitary_group.rvs(2**7), use_jax=True, check_arguments=False)
target = unitary_group.rvs(2**7)
# target = toffoli
gates_location = tuple([
    (1, 2),
    (0, 2),
    (1, 5),
    (0, 2),
    (0, 4),
    (4, 6),
    (0, 2),
    (5, 6),
    (0, 2),
    (1, 5),
    (0, 2),
    (0, 4),
    (4, 6),
    (0, 2),
    (1, 4),
])

amount_of_gates_in_cir = len(gates_location)
big_circuit = Circuit(7)
for loc in gates_location:
    big_circuit.append_gate(VariableUnitaryGate(2), loc)

circuit = big_circuit

method = 'qfactor_jax_batched_jit'
# method='qfactor_jax_batched'
# method='qfactor'
multistarts = 64
# %%
# Instantiate the circuit template with qfactor
tic = time.perf_counter()
circuit.instantiate(
    target,
    method=method,
    multistarts=multistarts,
    diff_tol_a=1e-12,   # Stopping criteria for distance change
    diff_tol_r=1e-6,    # Relative criteria for distance change
    dist_tol=1e-12,     # Stopping criteria for distance
    max_iters=10000,   # Maximum number of iterations
    min_iters=1000,     # Minimum number of iterations
    # slowdown_factor=0,   # Larger numbers slowdown optimization
    # to avoid local minima
)

toc = time.perf_counter()

print(f'Using {method} it took {toc-tic} seconeds for {multistarts}')

# Calculate and print final distance
dist = circuit.get_unitary().get_distance_from(target, 1)
print('Final Distance: ', dist)

# You can use synthesis to convert the `VariableUnitaryGate`s to
# native gates. Alternatively, you can build a circuit directly out of
# native gates and use the default instantiater to instantiate directly.

# %%
