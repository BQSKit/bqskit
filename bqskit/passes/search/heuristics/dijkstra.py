"""This module implements the DijkstraHeuristic class."""
from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


# TODO: Should name be changed to Breadth
class DijkstraHeuristic(HeuristicFunction):
    """
    The DijkstraHeuristic HeuristicFunction class.

    Defines a heuristic that relies only on circuit depth, which guarantees a
    minimal-depth final solution at the expense of a long runtime. This will
    create a behavior similar to breadth-first search.
    """

    def get_value(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Return the heuristic's value, see HeuristicFunction for more info."""
        cost = 0.0
        for gate in circuit.gate_set:
            if gate.num_qudits == 1:
                continue
            cost += float(circuit.count(gate))
        return cost
