import csv
import json
import time
import logging

from bqskit.compiler import Compiler

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate, CNOTGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.passes.synthesis.qsd import MGDPass
from bqskit.qis import UnitaryMatrix

from bqskit.passes.synthesis.bzxz import BlockZXZPass, FullBlockZXZPass
from bqskit.passes.synthesis.qsd import QSDPass

import numpy as np
from collections import Counter

def run_experiment(num_qudits: int, min_qudit_size: int, perform_extract: bool) -> dict[str, any]:
    # 1. Generate an n-qudit random Unitary Matrix 
    unitary = UnitaryMatrix.random(num_qudits)
    params = np.concatenate((np.real(unitary).flatten(), np.imag(unitary).flatten()))
    circuit = Circuit(num_qudits)

    # 2. Inserted them as a VU_nq gate into a circuit
    circuit.append_gate(VariableUnitaryGate(num_qudits), list(range(num_qudits)), params) 

    # 3. Copied workflow to synthesize (decompose) gates using FullBlockZXZ
    workflow = [
        FullBlockZXZPass(
            start_from_left=True,
            min_qudit_size=2,
            perform_extract=perform_extract
        ),
        QSDPass(),
    ]

    with Compiler(num_workers=1, runtime_log_level=logging.INFO) as compiler:
        start_time = time.time()
        compiled = compiler.compile(circuit, workflow)
        duration = time.time() - start_time

    # Construct unitary matrix from CNOT gate
    cnot_matrix = CNOTGate().get_unitary()

    # Used counter bc want to count and tally number of each gate type in circuit
    gate_counter = Counter()
    variable_unitary_2q = 0
    variable_unitary_3q = 0
    variable_unitary_4q = 0
    variable_unitary_5q = 0

    for op in compiled:
        # get specific gate
        g = op.gate
        if isinstance(g, VariableUnitaryGate):
            print(f"Gate: {type(g).__name__}, num_qudits: {getattr(g, 'num_qudits', 'N/A')}") 
        try:
            if isinstance(g, CNOTGate):
                gate_counter["Actual CNOT Count"] += 1
            if isinstance(g, VariableUnitaryGate):
                # num_qudits = g.num_qudits
                if g.num_qudits == 4:
                    # For n-qubit Variable Unitaries, (n-1) gates implemented by 2 CNOTs; 1 gate by 3 CNOTs
                    variable_unitary_4q += 1  # n-1 are 2-qubit gates
                elif g.num_qudits == 5:
                    variable_unitary_5q += 1
                elif g.num_qudits == 2:
                    variable_unitary_2q += 1  # 2-qubit gates
                elif g.num_qudits == 3:
                    variable_unitary_3q += 1  # 3-qubit gates
                else:
                    variable_unitary_2q += 0
        except Exception:
            gate_counter[type(g).__name__] += 1
        

    gate_counter.setdefault("Actual CNOT Count", 0)

    # Fidelity
    cost_fn = HilbertSchmidtResidualsGenerator()
    fidelity_cost = cost_fn(compiled, circuit.get_unitary())

    # print(np.linalg.norm(compiled.get_unitary() - circuit.get_unitary()))

    # theoretical lower bound using diagonalization – from paper
    if perform_extract:
        # Lower bound with diagonal extraction
        expected_CNOT_count = (9/16) * (4**num_qudits) - (3/2) * (2**num_qudits) + (5/3)
    else:
        # Upper bound or conservative estimate without diagonal extraction
        expected_CNOT_count = (9/16) * (4**num_qudits) - (3/2) * (2**num_qudits) + 10

    result = {
        "qubits": num_qudits,
        "min_qudit_size": min_qudit_size,
        "perform_extract": perform_extract,
        "compile_time_sec": round(duration, 4),
        "hilbert_cost": round(fidelity_cost, 8),
        "2 Qubit Variable Unitary Gates": variable_unitary_2q,
        "3 Qubit Variable Unitary Gates": variable_unitary_3q,
        "4 Qubit Variable Unitary Gates": variable_unitary_4q,
        "5 Qubit Variable Unitary Gates": variable_unitary_5q,
        "Expected CNOT Count": round(expected_CNOT_count, 2),
    }
    result.update(gate_counter)
    return result

def save_results_csv(results, filename='test_files/decompose_twice.csv'):
    if not results:
        return

    # Build unified set of all keys across all rows
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())
    all_keys = sorted(all_keys)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    # Testing on 4-6 qudits
    num_qudits = [3, 4, 5, 6]
    min_qudit_size = [2, 3, 4, 5]
    perform_extract_opts = [False, True]

    results = []
    total_jobs = len(num_qudits) * len(min_qudit_size) 
    # * len(perform_extract_opts)
    curr_job = 1

    # for pe in perform_extract_opts:
    for nq in num_qudits:
        for mqs in min_qudit_size:
            if nq <= 6 and mqs <= 5: # perform diagonal extraction when input size <= 6 and min_qudit_size <= 5
            # if nq <= 4 and mqs <= 3:
                pe = True
                print(f"[{curr_job}/{total_jobs}] Running: qudits={nq}, min_qudit_size={mqs}, extract={pe}")
                result = run_experiment(nq, mqs, pe)
                results.append(result)
            else:
                pe = False
                print(f"[{curr_job}/{total_jobs}] Running: qudits={nq}, min_qudit_size={mqs}, extract={pe}")
                result = run_experiment(nq, mqs, pe)
                results.append(result)
            curr_job += 1

    save_results_csv(results)
    print("\n✅ Results saved to decompose_twice.csv")
