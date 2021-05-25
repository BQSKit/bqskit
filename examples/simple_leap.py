"""This script is contains a simple use case of the QSearch synthesis method."""
from __future__ import annotations

import logging

import numpy as np

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler.passes.processing import ScanningGateRemovalPass
from bqskit.compiler.passes.synthesis import LEAPSynthesisPass
from bqskit.compiler.search.generators.simple import SimpleLayerGenerator
from bqskit.ir import Circuit
from bqskit.ir.gates import VariableUnitaryGate

# Enable logging
logging.getLogger('bqskit.compiler').setLevel(logging.DEBUG)

# Let's create a random 3-qubit unitary to synthesize and add it to a circuit.
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
circuit = Circuit.from_unitary(toffoli)

# We will now define the CompilationTask we want to run.
task = CompilationTask(
    circuit, [
        LEAPSynthesisPass(
            layer_generator=SimpleLayerGenerator(
                single_qudit_gate_1=VariableUnitaryGate(1),
            ),
            instantiate_options={
                'min_iters': 0,
                'diff_tol_r': 1e-5,
                'dist_tol': 1e-11,
                'max_iters': 2500,
            },
        ),
        ScanningGateRemovalPass(),
    ],
)

# Finally let's create create the compiler and execute the CompilationTask.
compiler = Compiler()
compiled_circuit = compiler.compile(task)
for op in compiled_circuit:
    print(op)

# Close our connection to the compiler backend
del compiler
