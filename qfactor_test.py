from bqskit.ir.gates.parameterized import MCRYGate, MCRZGate
import numpy as np
from sys import argv

# Nitrogen vacancy centers for 

num_qudits = int(argv[1])

mc = MCRZGate(num_qudits=num_qudits, controlled_qubit=(num_qudits - 1))

params = np.random.rand(2 ** (num_qudits - 1)) * np.pi

target = mc.get_unitary(params)

t_inv = target.dagger

out_params = mc.optimize(t_inv)

out_unitary = mc.get_unitary(out_params)

print(out_unitary.get_distance_from(target))