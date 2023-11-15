"""This script is contains a simple use case of the QFAST synthesis method."""
from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import ForEachBlockPass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix

# Let's create a random 3-qubit unitary to synthesize and add it to a
# circuit.
circuit = Circuit.from_unitary(UnitaryMatrix.random(3))

# We now define our synthesis workflow utilizing the QFAST algorithm.
workflow = [
    QFASTDecompositionPass(),
    ForEachBlockPass([
        LEAPSynthesisPass(),  # LEAP performs native gate instantiation
        ScanningGateRemovalPass(),  # Gate removal optimizing gate counts
    ]),
    UnfoldPass(),
]

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler() as compiler:
    compiled_circuit = compiler.compile(circuit, workflow)
    print(compiled_circuit.gate_counts)
