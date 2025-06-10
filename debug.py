import logging
import numpy as np
from numpy import allclose
from collections import Counter

from bqskit.passes import UnfoldPass
from bqskit.ir.gates import VariableUnitaryGate

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates.constant import CNOTGate
from bqskit.passes.synthesis.qsd import QSDPass
from bqskit.passes.synthesis.bzxz import FullBlockZXZPass
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator

# Create a 4-qubit random unitary
num_qudits = 4
unitary = UnitaryMatrix.random(num_qudits)
# circuit = Circuit.from_unitary(unitary)

params = np.concatenate((np.real(unitary).flatten(), np.imag(unitary).flatten()))
circuit = Circuit(num_qudits)
circuit.append_gate(VariableUnitaryGate(num_qudits), list(range(num_qudits)), params)

# Define synthesis pass with a small min_qudit_size to force full decomposition
workflow = [
    FullBlockZXZPass(
        start_from_left=True,
        min_qudit_size=2,      
        perform_extract=False   
    ),
    # QSDPass()
    UnfoldPass()
]

# Compile
with Compiler(num_workers=1, runtime_log_level=logging.INFO) as compiler:
    compiled = compiler.compile(circuit, workflow)

# Reference CNOT matrix
cnot_matrix = CNOTGate().get_unitary()

# Print and count gates
print("\n=== GATE BREAKDOWN ===")
gate_counter = Counter()
for op in compiled:
    gate = op.gate
    name = type(gate).__name__
    try:
        U = gate.get_unitary()
        if allclose(U, cnot_matrix):
            name = "CNOT"
    except Exception:
        pass
    gate_counter[name] += 1
    print(f"{name} on {gate.location if hasattr(gate, 'location') else 'unknown'}")

print("\nGate counts:")
for k, v in gate_counter.items():
    print(f"{k}: {v}")

# Check circuit fidelity
cost_fn = HilbertSchmidtResidualsGenerator()
cost = cost_fn(compiled, circuit.get_unitary())
print("\nHilbert-Schmidt cost:", cost)
