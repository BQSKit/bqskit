from bqskit.passes import FullQSDPass
from bqskit import Circuit
from bqskit.ir.gates import *
import numpy.random as rand
import numpy as np
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler
import time
from bqskit import enable_logging
from bqskitgpu.qfactor_jax import QFactor_jax
import logging

# QFactor hyperparameters - 
# see intantiation example for more detiles on the parameters
amount_of_workers = 10
num_multistarts = 32
max_iters = 100000
min_iters = 3
diff_tol_r = 1e-5
diff_tol_a = 0.0
dist_tol = 1e-10

diff_tol_step_r = 0.1
diff_tol_step = 200

beta = 0

# Preoare the instantiator
batched_instantiation = QFactor_jax(
                        diff_tol_r=diff_tol_r,
                        diff_tol_a=diff_tol_a,
                        min_iters=min_iters,
                        max_iters=max_iters,
                        dist_tol=dist_tol,
                        diff_tol_step_r=diff_tol_step_r,
                        diff_tol_step = diff_tol_step,
                        beta=beta)

instantiate_options={
'method': batched_instantiation,
'multistarts': num_multistarts,
}

enable_logging(True)

# Let's create a random 4-qubit unitary to synthesize and add it to a
# circuit.
circuit = Circuit.from_unitary(UnitaryMatrix.random(4))

# We now define our synthesis workflow utilizing the QFAST algorithm.
workflow = [
    FullQSDPass(start_from_left=True, min_qudit_size=2, instantiate_options=instantiate_options),
]

start = time.time()

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler(num_workers=amount_of_workers, runtime_log_level=logging.INFO) as compiler:
    start = time.time()
    compiled_circuit = compiler.compile(circuit, workflow)
    print(time.time() - start)
    print(compiled_circuit.gate_counts)

# For debugging

# QSDPass(min_qudit_size=2).run(circuit, None)

utry_1 = circuit.get_unitary()
utry_2 = compiled_circuit.get_unitary()

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(utry_1)
# print(utry_2)
# print(compiled_circuit)
print(utry_1.get_distance_from(utry_2))