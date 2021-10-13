"""This script is contains a simple use case of the QFAST synthesis method."""
from __future__ import annotations

import logging

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.processing import WindowOptimizationPass
from bqskit.passes.synthesis import LEAPSynthesisPass
from bqskit.passes.synthesis import QFASTDecompositionPass
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

if __name__ == '__main__':
    # Enable logging
    logging.getLogger('bqskit').setLevel(logging.DEBUG)

    # Let's create a random 3-qubit unitary to synthesize and add it to a
    # circuit.
    circuit = Circuit.from_unitary(UnitaryMatrix.random(3))

    # We will now define the CompilationTask we want to run.
    task = CompilationTask(
        circuit, [
            QFASTDecompositionPass(),
            ForEachBlockPass([LEAPSynthesisPass(), WindowOptimizationPass()]),
            UnfoldPass(),
        ],
    )

    # Finally let's create create the compiler and execute the CompilationTask.
    with Compiler() as compiler:
        compiled_circuit = compiler.compile(task)
