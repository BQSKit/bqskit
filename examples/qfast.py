"""This script is contains a simple use case of the QFAST synthesis method."""
from __future__ import annotations

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.passes import ForEachBlockPass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix

if __name__ == '__main__':
    # Let's create a random 3-qubit unitary to synthesize and add it to a
    # circuit.
    circuit = Circuit.from_unitary(UnitaryMatrix.random(3))

    # We will now define the CompilationTask we want to run.
    task = CompilationTask(
        circuit, [
            QFASTDecompositionPass(),
            ForEachBlockPass([LEAPSynthesisPass(), ScanningGateRemovalPass()]),
            UnfoldPass(),
        ],
    )

    # Finally let's create create the compiler and execute the CompilationTask.
    with Compiler() as compiler:
        compiled_circuit = compiler.compile(task)
        print(compiled_circuit)
