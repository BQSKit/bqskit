from bqskit.ir.gates.parameterized import MCRYGate, MCRZGate
import numpy as np
from sys import argv
import numpy as np
from bqskitrs import Circuit as CircuitRS
from bqskit.ir.circuit import Circuit

num_qudits = int(argv[1])

circuit = Circuit(num_qudits)

mc = MCRZGate(num_qudits=num_qudits, controlled_qubit=(num_qudits - 1))

circuit.append_gate(mc, list(range(num_qudits)))

crs = CircuitRS(circuit)

params = np.random.rand(2 ** (num_qudits - 1)) * np.pi

u1 = crs.get_unitary(params)

u2 = circuit.get_unitary(params)

# params = np.random.rand(2 ** (num_qudits - 1)) * np.pi

# target = mc.get_unitary(params)

# t_inv = target.dagger

# out_params = mc.optimize(t_inv)

# out_unitary = mc.get_unitary(out_params)

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=3)

# print(u1)
# print(u2)

print(u2.get_distance_from(u1))