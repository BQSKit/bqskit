import csv
import json
import time
import logging

from script_optimize_vus import synthesize_unoptimized_gates
from bqskit.compiler import Compiler

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate, CNOTGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.passes.synthesis.qsd import MGDPass
from bqskit.qis import UnitaryMatrix

from bqskit.passes.synthesis.bzxz import BlockZXZPass, FullBlockZXZPass
from bqskit.passes.synthesis.qsd import QSDPass

import asyncio
from bqskit.runtime import get_runtime
import numpy as np
from collections import Counter


# 1. Generate an n-qudit random Unitary Matrix 
def run_experiment(num_qudits: int, min_qudit_size: int, perform_extract: bool, decompose_all: bool) -> dict[str, any]:
    og_unitary = UnitaryMatrix.random(num_qudits)

    params = np.concatenate((np.real(og_unitary).flatten(), np.imag(og_unitary).flatten()))
    circuit = Circuit(num_qudits)

    # 2. Inserted them as a VU_nq gate into a circuit
    circuit.append_gate(VariableUnitaryGate(num_qudits), list(range(num_qudits)), params) 

    # 3. Copied workflow to synthesize (decompose) gates using FullBlockZXZ
    workflow = [
        # FullBlockZXZPass(
        #     start_from_left=True,
        #     min_qudit_size=min_qudit_size,
        #     perform_extract=perform_extract,
        #     decompose_all=decompose_all
        # )
        BlockZXZPass(
            min_qudit_size=min_qudit_size,
            decompose_all=decompose_all
        )
    ]

    with Compiler(num_workers=1, runtime_log_level=logging.INFO) as compiler:
        start_time = time.time()
        compiled = compiler.compile(circuit, workflow)
        duration = time.time() - start_time

        unoptimized_variable_unitaries = []
        for op in compiled:
            if isinstance(op.gate, VariableUnitaryGate) and op.gate.num_qudits > min_qudit_size:
                unoptimized_variable_unitaries.append((op.gate.num_qudits, op.location))

        if unoptimized_variable_unitaries:
            print(f"\nUnoptimized VariableUnitaryGates (Before Replacement): {len(unoptimized_variable_unitaries)}")
            for size, loc in unoptimized_variable_unitaries:
                print(f" - {size}-qubit VU at location {loc}")
        else:
            print("\nAll VariableUnitaryGates were decomposed!")

        # compiled = get_runtime().map(synthesize_unoptimized_gates, unoptimized_variable_unitaries)
        compiled = get_runtime().submit(synthesize_unoptimized_gates, unoptimized_variable_unitaries) # await 

        unoptimized_variable_unitaries = [(op.gate.num_qudits, op.location) for op in circuit if isinstance(op.gate, VariableUnitaryGate) and op.gate.num_qudits > min_qudit_size]
        num_remaining_vus = sum(1 for _ in unoptimized_variable_unitaries)
        print(f"\nUnoptimized VariableUnitaryGates (After Replacement): {num_remaining_vus}")
        if num_remaining_vus != 0:
            for size, loc in unoptimized_variable_unitaries:
                    print(f" - {size}-qubit VU at location {loc}")
        else:
            print(f" - \n")

        # --------------------------------------------------------------------------------------

        gate_counter = Counter()
        variable_unitary_2q = 0
        variable_unitary_3q = 0
        variable_unitary_4q = 0
        variable_unitary_5q = 0

        for op in compiled:
            g = op.gate # get specific gate
            try:
                if isinstance(g, CNOTGate):
                    gate_counter["Actual CNOT Count"] += 1
                if isinstance(g, VariableUnitaryGate):
                    if g.num_qudits == 2:
                        variable_unitary_2q += 1 
                    elif g.num_qudits == 3:
                        variable_unitary_3q += 1 
                    elif g.num_qudits == 4:
                        variable_unitary_4q += 1
                    elif g.num_qudits == 5:
                        variable_unitary_5q += 1
                    else:
                        variable_unitary_2q += 0
            except Exception:
                gate_counter[type(g).__name__] += 1

        gate_counter.setdefault("Actual CNOT Count", 0)

        result = {
            "decompose_all": True,
            "qubits": num_qudits,
            "min_qudit_size": min_qudit_size,
            "perform_extract": perform_extract,
            "2 Qubit Variable Unitaries": variable_unitary_2q,
            "3 Qubit Variable Unitaries": variable_unitary_3q,
            "4 Qubit Variable Unitaries": variable_unitary_4q,
            "5 Qubit Variable Unitaries": variable_unitary_5q,
            # "CNOT TLB": round(CNOT_Theoretical_Lower_Bound, 2),
            "compile_time_sec": round(duration, 4),
            "hilbert_cost": "N/A",
        }
        result.update(gate_counter)

        print(f"\n--------- New Gate Summary ---------")
        print(f" - 2-Qubit VariableUnitaryGates: {result['2 Qubit Variable Unitaries']}")
        print(f" - 3-Qubit VariableUnitaryGates: {result['3 Qubit Variable Unitaries']}")
        print(f" - 4-Qubit VariableUnitaryGates: {result['4 Qubit Variable Unitaries']}")
        print(f" - 5-Qubit VariableUnitaryGates: {result['5 Qubit Variable Unitaries']}")
        print(f" - Actual CNOT Count: {gate_counter['Actual CNOT Count']}\n")

    # Construct unitary matrix from CNOT gate
    # print(f"\n - Original Unitary Dimensions: {og_unitary.shape}")
    # print(f"H-S Norm (Before): {np.linalg.norm(compiled.get_unitary() - og_unitary):.2e}\n")

    # Used counter bc want to count and tally number of each gate type in circuit
    gate_counter = Counter()
    variable_unitary_2q = 0
    variable_unitary_3q = 0
    variable_unitary_4q = 0
    variable_unitary_5q = 0

    unoptimized_variable_unitaries = []

    for op in compiled:
        g = op.gate # get specific gate
        try:
            if isinstance(g, CNOTGate):
                gate_counter["Actual CNOT Count"] += 1
            if isinstance(g, VariableUnitaryGate):
                if g.num_qudits == 2:
                    variable_unitary_2q += 1 
                elif g.num_qudits == 3:
                    variable_unitary_3q += 1 
                elif g.num_qudits == 4:
                    variable_unitary_4q += 1
                elif g.num_qudits == 5:
                    variable_unitary_5q += 1
                else:
                    variable_unitary_2q += 0
        except Exception:
            gate_counter[type(g).__name__] += 1

    gate_counter.setdefault("Actual CNOT Count", 0)

    # Fidelity
    cost_fn = HilbertSchmidtResidualsGenerator()
    fidelity_cost = cost_fn(compiled, og_unitary)


    # theoretical lower bound using diagonalization - from paper
    if perform_extract:
        # Lower bound with diagonal extraction
        fidelity_check = "Since diagonals were extracted and commuted during D.E., a direct fidelity comparison is not meaningful"
        CNOT_Theoretical_Lower_Bound = (9/16) * (4**num_qudits) - (3/2) * (2**num_qudits) + (5/3)
    else:
        # Upper bound or conservatirve estimate without diagonal extraction
        fidelity_check = {np.linalg.norm(compiled.get_unitary() - og_unitary):.2e}
        CNOT_Theoretical_Lower_Bound = (9/16) * (4**num_qudits) - (3/2) * (2**num_qudits) + 10
    
    print(f"\n--------- Fidelity Check ---------")
    print(f" - Original Unitary Dimensions: {og_unitary.shape}")
    print(f" - Compiled Unitary Dimensions: {compiled.get_unitary().shape}")
    # print(f"H-S Norm (After): {np.linalg.norm(compiled.get_unitary() - og_unitary):.2e}")
    if isinstance(fidelity_check, str):
        print(f" - Note: {fidelity_check}")
    else:
        print(f" - Fidelity (np.allclose): {fidelity_check}")
    
    result = {
        "decompose_all": True,
        "qubits": num_qudits,
        "min_qudit_size": min_qudit_size,
        "perform_extract": perform_extract,
        "2 Qubit Variable Unitaries": variable_unitary_2q,
        "3 Qubit Variable Unitaries": variable_unitary_3q,
        "4 Qubit Variable Unitaries": variable_unitary_4q,
        "5 Qubit Variable Unitaries": variable_unitary_5q,
        # "CNOT TLB": round(CNOT_Theoretical_Lower_Bound, 2),
        "compile_time_sec": round(duration, 4),
        "hilbert_cost": fidelity_cost,
    }
    result.update(gate_counter)

    print(f"\n--------- Old Gate Summary ---------")
    print(f" - 2-Qubit VariableUnitaryGates: {result['2 Qubit Variable Unitaries']}")
    print(f" - 3-Qubit VariableUnitaryGates: {result['3 Qubit Variable Unitaries']}")
    print(f" - 4-Qubit VariableUnitaryGates: {result['4 Qubit Variable Unitaries']}")
    print(f" - 5-Qubit VariableUnitaryGates: {result['5 Qubit Variable Unitaries']}")
    print(f" - Actual CNOT Count: {gate_counter['Actual CNOT Count']}")

    print(f"\n--------- Timing ---------")
    print(f"\n⏱ Compile Time: {duration:.2f} seconds\n")
    return result

def save_results_csv(results, filename='test_files/decompose_parallel_test.csv'):
    if not results:
        return
    
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())
    all_keys = [
        "qubits",
        "min_qudit_size",
        "perform_extract",
        "decompose_all",
        "Actual CNOT Count",
        "compile_time_sec",
        "hilbert_cost",
        "2 Qubit Variable Unitaries",
        "3 Qubit Variable Unitaries",
        "4 Qubit Variable Unitaries",
        "5 Qubit Variable Unitaries",
        # "CNOT TLB"
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    results = []

    nq = 6
    # mqs = [2, 3, 4, 5]
    mq = 2
    pe = [False, True]

    curr_job = 1
    # total_jobs = len(perform_extract_opts) * len(perform_extract_opts) * len(mqs)
    total_jobs = len(pe) * len(pe)

    # for mq in mqs:
    #     for da in pe:
    #         for p in pe:
    #             print(f"[{curr_job}/{total_jobs}] Running BZXZ Decomposition: qubits={nq}, min_qudit_size={mq}, perform_extract={p}, decompose_all={da}")
    #             res = run_experiment(nq, mq, p, da)
    #             results.append(res)
    #             curr_job += 1


    print(f"\n[{curr_job}/1] Running BZXZ Decomposition: qubits={nq}, min_qudit_size={mq}, perform_extract={False}, decompose_all={True}\n")
    res = run_experiment(nq, min_qudit_size=mq, perform_extract=False, decompose_all=True)
    results.append(res)
    curr_job += 1


    save_results_csv(results)
    print("\n✅ Results saved to decompose_all.csv")