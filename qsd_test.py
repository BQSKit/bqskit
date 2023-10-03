from bqskit.passes import FullQSDPass
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.gates import *
import numpy.random as rand
import numpy as np
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler
import time
from bqskit import enable_logging
# from bqskitqfactorjax.qfactor_jax import QFactor_jax
import logging
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
import jax.config as config
from sys import argv
import pickle
# from bqskit import compile

# config.update("jax_enable_x64",True)

# # QFactor hyperparameters - 
# # see intantiation example for more detiles on the parameters
amount_of_workers = 2
# num_multistarts = 32
# max_iters = 100000
# min_iters = 3
# diff_tol_r = 1e-5
# diff_tol_a = 0.0
# dist_tol = 1e-8

# diff_tol_step_r = 0.1
# diff_tol_step = 200

# beta = 0

# # Preoare the instantiator
# batched_instantiation = QFactor_jax(
#                         diff_tol_r=diff_tol_r,
#                         diff_tol_a=diff_tol_a,
#                         min_iters=min_iters,
#                         max_iters=max_iters,
#                         dist_tol=dist_tol,
#                         diff_tol_step_r=diff_tol_step_r,
#                         diff_tol_step = diff_tol_step,
#                         beta=beta)

# instantiate_options={
# 'method': batched_instantiation,
# 'multistarts': num_multistarts,
# }
instantiate_options = {'min_iters': 0,
                       'diff_tol_r':1e-4}

# enable_logging(True)

# Let's create a random 4-qubit unitary to synthesize and add it to a
# circuit.
num_qudits = int(argv[2])
circ_type = argv[1]
min_qudits = int(argv[3]) if len(argv) == 4 else 2
circuit = Circuit(num_qudits)

ccx_unitary = np.identity(2 ** num_qudits)
ccx_unitary[-1, -1] = 0
ccx_unitary[-1, -2] = 1
ccx_unitary[-2, -1] = 1
ccx_unitary[-2, -2] = 0

# print(ccx_unitary)

if circ_type == "random":
    unitary = UnitaryMatrix.random(num_qudits)
elif circ_type == "ccx":
    unitary = ccx_unitary
elif circ_type == "qft":
    unitary = pickle.load(open(f"unitaries/qft_{num_qudits}.unitary", "rb"))

for _ in range(1):
    circ = Circuit.from_unitary(ccx_unitary)
    circuit.append_circuit(circ, list(range(num_qudits)))
    # circuit.append_gate(CNOTGate(), (num_qudits - 1, num_qudits))

# We now define our synthesis workflow utilizing the QFAST algorithm.
workflow = [
    FullQSDPass(start_from_left=True, min_qudit_size=min_qudits, instantiate_options=instantiate_options),
]

start = time.time()

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler(num_workers=amount_of_workers, runtime_log_level=logging.INFO) as compiler:
    start = time.time()
    compiled_circuit = compiler.compile(circuit, workflow)
    print(time.time() - start)
    print(dict(sorted(compiled_circuit.gate_counts.items(), key=lambda x: x[0].name)))
# compiled_circuit = compile(circuit, optimization_level=4, max_synthesis_size=4)

utry_1 = compiled_circuit.get_unitary()
utry_2 = circuit.get_unitary()

cost_function = HilbertSchmidtResidualsGenerator()
print(cost_function(compiled_circuit, circuit.get_unitary()))

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(utry_1)
# print(utry_2)
# print(compiled_circuit)
# print(utry_1.get_distance_from(utry_2))