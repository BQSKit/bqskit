"""This module implements MachineModels for Rigetti's QPUs."""
from __future__ import annotations

from bqskit.compiler.machine import MachineModel
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SXGate
from bqskit.ir.gates import XGate
from bqskit.qis.graph import CouplingGraph

rigetti_gate_set = {SXGate(), XGate(), RZGate(), CZGate()}

_aspen_11_coupling_graph = CouplingGraph([
    # Ring 1
    (0, 1), (1, 2), (2, 3), (3, 4),
    (4, 5), (5, 6), (6, 7), (7, 0),

    # Ring 2
    (8, 9), (9, 10), (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15), (15, 8),

    # Ring 3
    (16, 17), (17, 18), (19, 20),
    (20, 21), (21, 22), (22, 23), (23, 16),

    # Ring 4
    (24, 25), (25, 26), (26, 27), (27, 28),
    (28, 29), (29, 30), (30, 31), (31, 24),

    # Ring 5
    (34, 35), (35, 36),
    (36, 37), (37, 38), (38, 39),

    # Ring 1-2
    (1, 14), (2, 13),

    # Ring 2-3
    (9, 22), (10, 21),

    # Ring 3-4
    (17, 30),

    # Ring 4-5
    (25, 38), (26, 37),
])
"""Retrieved August 31, 2022: https://qcs.rigetti.com/qpus."""

_octo_rings = [
    [
        (i + 0, i + 1), (i + 1, i + 2), (i + 2, i + 3), (i + 3, i + 4),
        (i + 4, i + 5), (i + 5, i + 6), (i + 6, i + 7), (i + 7, i + 0),
    ]
    for i in range(0, 10 * 8, 8)
]

_horizontal_connections = [
    [
        (i + 1, j + 6),
        (i + 2, j + 5),
    ]
    for i, j in [(8 * r, 8 * (r + 1)) for r in [0, 1, 2, 3, 5, 6, 7, 8]]
]

_vertical_connections = [
    [
        (i + 0, j + 3),
        (i + 7, j + 4),
    ]
    for i, j in [(8 * r, 8 * (r + 5)) for r in range(5)]
]
_links = []
for l in _octo_rings:
    _links.extend(l)
for l in _horizontal_connections:
    _links.extend(l)
for l in _vertical_connections:
    _links.extend(l)

_aspen_m2_coupling_graph = CouplingGraph(_links)
"""Retrieved August 31, 2022: https://qcs.rigetti.com/qpus."""

Aspen11Model = MachineModel(40, _aspen_11_coupling_graph, rigetti_gate_set)
"""A BQSKit MachineModel for Rigetti's Aspen-11 quantum processor."""

AspenM2Model = MachineModel(80, _aspen_m2_coupling_graph, rigetti_gate_set)
"""A BQSKit MachineModel for Rigetti's Aspen-M-2 quantum processor."""

__all__ = ['Aspen11Model', 'AspenM2Model']
