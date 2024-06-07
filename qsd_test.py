from bqskit.compiler import Compiler #, compile
from bqskit.passes import FullQSDPass
from bqskit.ir.circuit import Circuit
import numpy as np
from bqskit.qis import UnitaryMatrix
import time
from sys import argv
import pickle

# config.update("jax_enable_x64",True)

# # QFactor hyperparameters - 
# # see intantiation example for more detiles on the parameters
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
circ_type = argv[1]
num_qudits = int(argv[2])
min_qudits = int(argv[3])
tree_depth = int(argv[4])
amount_of_workers = int(argv[5])
log_file = argv[6] if len(argv) == 7 else None
partition_depth = None
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
    circ = Circuit.from_unitary(unitary)
    circuit.append_circuit(circ, list(range(num_qudits)))
    # circuit.append_gate(CNOTGate(), (num_qudits - 1, num_qudits))

# circuit: Circuit = pickle.load(open("qft4noscan.pickle", "rb"))

# partition_depth = None
# tree_depth = 2


# We now define our synthesis workflow utilizing the QFAST algorithm.
workflow = [
    FullQSDPass(start_from_left=True, min_qudit_size=min_qudits, instantiate_options=instantiate_options, tree_depth=tree_depth,partition_depth=partition_depth),
]

start = time.time()

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler(num_workers=amount_of_workers, log_file=log_file) as compiler:
    start = time.time()
    compiled_circuit = compiler.compile(circuit, workflow)
    total_time = time.time() - start
    # compiled_circuit = compile(compiled_circuit, optimization_level=4, max_synthesis_size=3, compiler=compiler)
    # print(time.time() - start)

gates = sorted(compiled_circuit.gate_counts.items(), key=lambda x: x[0].name)

print(dict(gates))

# gates = [x[1] for x in gates]


# print([type(x) for x in compiled_circuit.gate_counts.keys()])


# scan_type = f"treescan{tree_depth}"
# pickle.dump(compiled_circuit, open(f"{circ_type}_{num_qudits}_{scan_type}_{partition_depth}.pickle", "wb"))


# utry_1 = compiled_circuit.get_unitary()
# utry_2 = circuit.get_unitary()

# cost_function = HilbertSchmidtResidualsGenerator()
# print(cost_function(compiled_circuit, circuit.get_unitary()))

# print(f"{circ_type}_{scan_type}, {num_qudits}, {partition_depth}, {total_time}, {gates[0]}, {gates[1]}, {gates[2]}, {gates[3]}")


# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(utry_1)
# print(utry_2)
# print(compiled_circuit)
# print(utry_1.get_distance_from(utry_2))