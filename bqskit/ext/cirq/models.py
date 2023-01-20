"""This module implements pre-built models for Google's QPUs."""
from __future__ import annotations

from bqskit.compiler.machine import MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import PhasedXZGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SqrtISwapGate
from bqskit.ir.gates import SycamoreGate
from bqskit.qis.graph import CouplingGraph

google_gate_set: set[Gate] = {
    PhasedXZGate(),
    RZGate(),
    SycamoreGate(),
    CZGate(),
    SqrtISwapGate(),
}

# Sycamore Device
_edges = [
    (20, 14), (34, 26), (35, 20), (8, 3), (8, 17), (47, 16),
    (36, 28), (48, 28), (43, 38), (38, 14), (12, 6), (47, 12),
    (46, 31), (48, 11), (24, 18), (30, 52), (24, 33), (0, 50),
    (8, 33), (45, 14), (34, 18), (7, 31), (48, 44), (19, 13), (0, 25),
    (19, 42), (1, 50), (44, 21), (6, 52), (1, 44), (7, 39), (29, 19),
    (2, 23), (14, 23), (41, 35), (15, 38), (19, 6), (51, 4), (0, 32),
    (36, 53), (48, 53), (41, 6), (47, 41), (29, 49), (35, 52), (22, 8),
    (9, 31), (51, 37), (26, 10), (1, 25), (35, 23), (2, 45), (0, 49),
    (50, 21), (43, 20), (16, 3), (15, 39), (47, 34), (49, 42), (51, 18),
    (5, 32), (26, 20), (37, 31), (45, 27), (22, 16), (12, 21), (43, 10),
    (24, 3), (41, 26), (22, 44), (13, 52), (29, 50), (30, 23), (18, 10),
    (5, 49), (29, 12), (24, 4), (43, 39), (46, 10), (17, 28), (34, 3),
    (38, 27), (16, 21), (45, 40), (1, 11), (22, 28), (46, 39), (51, 46)
]
_sycamore_coupling_graph = CouplingGraph(_edges)
SycamoreModel = MachineModel(54, _sycamore_coupling_graph, google_gate_set)

# Sycamore23 Device
_edges = [
    (20, 3), (18, 12), (8, 12), (4, 18), (14, 8), (6, 2), (15, 2),
    (20, 7), (21, 9), (4, 19), (0, 16), (14, 18), (13, 10), (20, 17),
    (21, 18), (7, 2), (16, 22), (16, 12), (5, 2), (14, 9), (1, 19),
    (17, 9), (14, 10), (15, 9), (8, 22), (5, 10), (0, 19), (12, 19),
    (16, 11), (13, 8), (15, 10), (20, 15)
]
_sycamore23_coupling_graph = CouplingGraph(_edges)
Sycamore23Model = MachineModel(23, _sycamore23_coupling_graph, google_gate_set)

__all__ = ['SycamoreModel', 'Sycamore23Model']
