from bqskit.passes import FullQSDPass
from bqskit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.gates import *
import numpy.random as rand
import numpy as np
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler
import time
from bqskit import enable_logging
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator


# enable_logging(True)

# Let's create a random 4-qubit unitary to synthesize and add it to a
# circuit.
start_num = 4
end_num = 2

circuit = Circuit(num_qudits=start_num)
for _ in range(2):
    circ = Circuit.from_unitary(UnitaryMatrix.random(start_num))
    # circ.append_gate(CNOTGate(), (0,2))
    circuit.append_circuit(circ, list(range(start_num)))

# We now define our synthesis workflow utilizing the QFAST algorithm.
workflow = [
    FullQSDPass(min_qudit_size=end_num),
]

start = time.time()

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler() as compiler:
    start = time.time()
    compiled_circuit = compiler.compile(circuit, workflow)
    print(time.time() - start)
    print(compiled_circuit.gate_counts)

# For debugging

# QSDPass(min_qudit_size=2).run(circuit, None)

# np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=3)
# for cycle, op in compiled_circuit.operations_with_cycles():
#     print(op.get_unitary())

utry_1 = compiled_circuit.get_unitary()
utry_2 = circuit.get_unitary()

cost_function = HilbertSchmidtResidualsGenerator()
print(cost_function(compiled_circuit, circuit.get_unitary()))

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(utry_1)
# print(utry_2)
# print(compiled_circuit)
print(utry_1.get_distance_from(utry_2))