"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

from bqskit import compile, Circuit
from bqskit.passes import *
from bqskit.compiler import Compiler
import sys
import time

# Construct the unitary as an NumPy array
circ = Circuit.from_file("tfxy_5.qasm")
circ.remove_all_measurements()

print(circ.gate_counts)

task = sys.argv[1]
block_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
# The compile function will perform synthesis

# Create a controlled workflow. 

# Model: Log to Timeline

compiler = Compiler(num_workers=num_workers, run_profiler=True)

start_time = time.time()

# synthesized_circuit = compile(circ, max_synthesis_size=3, optimization_level=3, compiler=compiler)
if task == "leap":
    workflow = [
        ScanPartitioner(block_size),
        ForEachBlockPass([
            LEAPSynthesisPass(),  # LEAP performs native gate instantiation
        ]),
        UnfoldPass(),
    ]
elif task == "scan":
    workflow = [
        ScanPartitioner(block_size),
        ForEachBlockPass([
            ScanningGateRemovalPass(),  # Gate removal optimizing gate counts
        ]),
        UnfoldPass(),
    ]
elif task == "qsearch":
    workflow = [
        ScanPartitioner(block_size),
        ForEachBlockPass([
            QSearchSynthesisPass(),  # QSearch Synthesis Pass
        ]),
        UnfoldPass(),
    ]
elif task == "pas":
    workflow = [
        ScanPartitioner(block_size),
        ForEachBlockPass([
            PermutationAwareSynthesisPass(),
        ]),
        UnfoldPass(),
    ]

synthesized_circuit = compiler.compile(circ, workflow=workflow)

print(synthesized_circuit.gate_counts)
print("Total Time:", time.time() - start_time)
