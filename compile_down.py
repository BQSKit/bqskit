from bqskit.passes import FullQSDPass
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.gates import *
import numpy.random as rand
import numpy as np
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler, compile
import time
from bqskit import enable_logging
# from bqskitqfactorjax.qfactor_jax import QFactor_jax
import logging
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
import jax.config as config
from sys import argv
import pickle
from bqskit import compile

in_circ = pickle.load(open("ccx4scan.pickle", "rb"))

# Finally let's create create the compiler and execute the CompilationTask.
with Compiler(num_workers=32, runtime_log_level=logging.INFO) as compiler:
    compiled_circuit = compile(in_circ, optimization_level=4, compiler=compiler)
    # compiled_circuit = compile(compiled_circuit, optimization_level=4, max_synthesis_size=3, compiler=compiler)
    # print(time.time() - start)

print(dict(sorted(compiled_circuit.gate_counts.items(), key=lambda x: x[0].name)))

pickle.dump(compiled_circuit, open("ccx4scan.pickle", "wb"))


utry_1 = compiled_circuit.get_unitary()
utry_2 = in_circ.get_unitary()

cost_function = HilbertSchmidtResidualsGenerator()
print(cost_function(compiled_circuit, in_circ.get_unitary()))

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(utry_1)
# print(utry_2)
# print(compiled_circuit)
# print(utry_1.get_distance_from(utry_2))