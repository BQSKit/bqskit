from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import *
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler
import time
import logging
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.passes.processing import ScanningGateRemovalPass, TreeScanningGateRemovalPass
from bqskit.passes.util import ToVariablePass
import pickle
from bqskit import enable_logging
import numpy as np

# enable_logging(True)

amount_of_workers = 8
# circuit: Circuit = pickle.load(open("qft4noscan.pickle", "rb"))
circuit = Circuit(5)
for i in range(4):
    circuit.append_gate(U3Gate(), (1,), np.random.rand(3) * np.pi)
    circuit.append_gate(U3Gate(), (2,), np.random.rand(3) * np.pi)
    circuit.append_gate(U3Gate(), (3,), np.random.rand(3) * np.pi)
    circuit.append_gate(U3Gate(), (0,), np.random.rand(3) * np.pi)
    circuit.append_gate(CNOTGate(), (0, 2))
    circuit.append_gate(CNOTGate(), (1, 3))
    circuit.append_gate(CNOTGate(), (1, 3))
    circuit.append_gate(XGate(), (0,))


instantiate_options = {'min_iters': 0,
                       'diff_tol_r':1e-4,
                       "method":"qfactor"}

scan = ScanningGateRemovalPass(instantiate_options=instantiate_options)
tscan = TreeScanningGateRemovalPass(tree_depth=4, instantiate_options=instantiate_options)

workflow_scan = [
    ToVariablePass(),
    scan
]

# tscan.run(circuit=circuit, data=None)


start = time.time()

workflow_tscan = [
    ToVariablePass(),
    tscan
]


print(dict(sorted(circuit.gate_counts.items(), key=lambda x: x[0].name)))

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler(num_workers=amount_of_workers, runtime_log_level=logging.INFO) as compiler:
    start = time.time()
    compiled_circuit = compiler.compile(circuit, workflow_scan)
    print(time.time() - start)
    print(dict(sorted(circuit.gate_counts.items(), key=lambda x: x[0].name)))
    start = time.time()
    compiled_circuit_2 = compiler.compile(circuit, workflow_tscan)
    print(time.time() - start)

print(dict(sorted(compiled_circuit.gate_counts.items(), key=lambda x: x[0].name)))
print(dict(sorted(compiled_circuit_2.gate_counts.items(), key=lambda x: x[0].name)))
# pickle.dump(compiled_circuit, open("qft4postscan.pickle", "wb"))

cost_function = HilbertSchmidtResidualsGenerator()
print(cost_function(compiled_circuit, circuit.get_unitary()))
print(cost_function(compiled_circuit_2, circuit.get_unitary()))
