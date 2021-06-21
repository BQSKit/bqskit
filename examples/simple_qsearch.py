"""This script is contains a simple use case of the QSearch synthesis method."""
from __future__ import annotations

import logging

import numpy as np

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler.passes.synthesis import QSearchSynthesisPass
from bqskit.ir import Circuit

# Enable logging
logging.getLogger('bqskit.compiler').setLevel(logging.DEBUG)

# Let's create a 3-qubit toffoi unitary to synthesize and add it to a circuit.
toffoli = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype='complex128',
)

circuit = Circuit.from_unitary(toffoli)

# We will now define the CompilationTask we want to run.
task = CompilationTask(circuit, [QSearchSynthesisPass(success_threshold=1e-9)])

# Finally let's create create the compiler and execute the CompilationTask.
compiler = Compiler()
compiled_circuit = compiler.compile(task)
for op in compiled_circuit:
    print(op)

# Close our connection to the compiler backend
del compiler
