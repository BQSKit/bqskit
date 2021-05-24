"""This script is contains a simple use case of the QFAST synthesis method."""
from __future__ import annotations

import logging

from scipy.stats import unitary_group

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler.passes.synthesis import QFASTDecompositionPass
from bqskit.ir import Circuit

# Enable logging
logging.getLogger('bqskit').setLevel(logging.DEBUG)

# Let's create a random 3-qubit unitary to synthesize and add it to a circuit.
circuit = Circuit.from_unitary(unitary_group.rvs(8))

# We will now define the CompilationTask we want to run.
task = CompilationTask(circuit, [QFASTDecompositionPass()])

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler() as compiler:
    compiled_circuit = compiler.compile(task)
    for op in compiled_circuit:
        print(op)
