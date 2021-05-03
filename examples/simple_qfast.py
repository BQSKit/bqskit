"""This script is contains a simple use case of the QFAST synthesis method."""

from bqskit.ir import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler import CompilationTask
from bqskit.compiler.passes.synthesis import QFASTDecompositionPass

# Enable logging
import logging
logging.getLogger("bqskit").setLevel(logging.DEBUG)

# Let's create a random 3-qubit unitary to synthesize and add it to a circuit.
from scipy.stats import unitary_group
circuit = Circuit.from_unitary(unitary_group.rvs(8))

# We will now define the CompilationTask we want to run.
task = CompilationTask(circuit, [QFASTDecompositionPass()])

# Finally let's create create the compiler and execute the CompilationTask.
compiler = Compiler()
compiled_circuit = compiler.compile(task)
for op in compiled_circuit:
    print(op)

# Close our connection to the compiler backend
del compiler