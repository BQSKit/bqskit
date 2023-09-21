from bqskit.ir.gates.parameterized import MCRYAccGate, MCRZAccGate
from bqskit.qis.unitary import UnitaryMatrix
from bqskit import enable_logging
import numpy as np
import jax.numpy as jnp
from sys import argv
import pdb
import jax.config as config

enable_logging(True)
config.update("jax_enable_x64",True)

# Nitrogen vacancy centers for 

num_qudits = int(argv[1])

mc = MCRYAccGate(num_qudits=num_qudits)
params = np.random.rand(2 ** (num_qudits - 1)) * np.pi
target = mc.get_unitary(params)

t_inv = target.dagger

out_params = mc.optimize(t_inv._utry)

out_unitary = mc.get_unitary(out_params)

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=3)
out_utry = np.array(out_unitary._utry, dtype=np.complex128)

np_out = UnitaryMatrix.closest_to(out_utry)
np_target = UnitaryMatrix.closest_to(np.array(target._utry, dtype=np.complex128))

print(np_out.get_distance_from(np_target))