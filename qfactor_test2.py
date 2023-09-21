from bqskit import Circuit
import numpy as np
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates.parameterized import MCRYGate, MCRZGate


num_qudits = 3

target = UnitaryMatrix.random(num_qudits=num_qudits)

circuit = Circuit.from_unitary(target)

mc = MCRZGate(3, 2)
params = np.random.rand(4) * np.pi

circuit.insert_gate(1, mc, list(range(3)), params)

env_matrix = circuit.get_unitary().dagger

out_params = mc.optimize(env_matrix)

out_utry = mc.get_unitary(out_params)

print(target.get_distance_from(out_utry))
print(target.get_distance_from(mc.get_unitary(params)))
