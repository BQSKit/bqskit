"""This script is contains a simple use case of the QSearch synthesis method."""
from __future__ import annotations
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.gates.constant.iswap import ISwapGate
from bqskit.compiler.search.generators.simple import SimpleLayerGenerator

import logging

import numpy as np

from scipy.stats import unitary_group

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler.passes.synthesis import QSearchSynthesisPass
from bqskit.ir import Circuit

from bqskitrs import HilbertSchmidtCostFunction
from bqskitrs import LBFGSMinimizerNative

from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.function import CostFunction
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.instantiaters.minimization import Minimization

class NativeMinimizer(LBFGSMinimizerNative, Minimizer):
    ...

class NativeCostFunction(HilbertSchmidtCostFunction, DifferentiableCostFunction):
    ...

class NativeGenerator(CostFunctionGenerator):
    def gen_cost(self, circuit: Circuit, target: UnitaryMatrix | StateVector) -> CostFunction:
        return NativeCostFunction(circuit, target.get_numpy()) 



# Enable logging
logging.getLogger('bqskit.compiler.*').setLevel(logging.DEBUG)

toffoli = np.array([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0]], 
                     dtype='complex128')

# Let's create a random 3-qubit unitary to synthesize and add it to a circuit.
circuit = Circuit.from_unitary(toffoli)

# We will now define the CompilationTask we want to run.
task = CompilationTask(circuit, [QSearchSynthesisPass(success_threshold=1e-9, cost=NativeGenerator(), instantiate_options={'minimizer': NativeMinimizer()})])

# Finally let's create create the compiler and execute the CompilationTask.
compiler = Compiler()
compiled_circuit = compiler.compile(task)


# Close our connection to the compiler backend
del compiler
