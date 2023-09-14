from bqskit.passes import QSDPass
from bqskit import Circuit
from bqskit.ir.gates import *
import numpy.random as rand
import numpy as np
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler
import time


# Let's create a random 4-qubit unitary to synthesize and add it to a
# circuit.
circuit = Circuit.from_unitary(UnitaryMatrix.random(10))

# We now define our synthesis workflow utilizing the QFAST algorithm.
workflow = [
    QSDPass(min_qudit_size=2),
]

start = time.time()

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler() as compiler:
    compiled_circuit = compiler.compile(circuit, workflow)
    print(compiled_circuit.gate_counts)

print(time.time() - start)

utry_1 = circuit.get_unitary()
utry_2 = compiled_circuit.get_unitary()

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(utry_1)
# print(utry_2)
# print(compiled_circuit)
print(utry_1.get_distance_from(utry_2))