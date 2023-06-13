"""
This module implements circuit metrics used in the Supermarq benchmark suite.

References:
    Tomesh, Teague, et al. "Supermarq: A scalable quantum benchmark suite."
    2022 IEEE International Symposium on High-Performance Computer
    Architecture (HPCA). IEEE, 2022.
"""
from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit


def supermarq_program_communication(circuit: Circuit) -> float:
    """SupermarQ program communication metric."""
    n = circuit.num_qudits
    degrees_sum = sum(circuit.coupling_graph.get_qudit_degrees())
    return degrees_sum / (n * (n - 1))


def supermarq_critical_depth(circuit: Circuit) -> float:
    """SupermarQ critical depth metric."""
    qudit_depths = np.zeros(circuit.num_qudits, dtype=int)
    num_multi_qubit_gates = 0
    for op in circuit:
        if len(op.location) > 1:
            new_depth = max(qudit_depths[list(op.location)]) + 1
            qudit_depths[list(op.location)] = new_depth
            num_multi_qubit_gates += 1
    return int(max(qudit_depths)) / num_multi_qubit_gates


def supermarq_entanglement_ratio(circuit: Circuit) -> float:
    """SupermarQ entanglement-ratio metric."""
    num_multi_qubit_gates = 0
    num_gates = 0
    for op in circuit:
        num_gates += 1
        if len(op.location) > 1:
            num_multi_qubit_gates += 1
    return num_multi_qubit_gates / num_gates


def supermarq_parallelism(circuit: Circuit) -> float:
    """SupermarQ parallelism metric."""
    d = circuit.depth
    ng = circuit.num_operations
    return (ng / d - 1) / (circuit.num_qudits - 1)


def supermarq_liveness(circuit: Circuit) -> float:
    """SupermarQ liveness metric."""
    liveness_sum = 0
    for i in range(circuit.num_qudits):
        for j in range(circuit.num_cycles):
            if circuit._circuit[j][i] is not None:
                liveness_sum += 1
    return liveness_sum / (circuit.num_qudits * circuit.depth)
