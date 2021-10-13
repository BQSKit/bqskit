"""This module implements the RecordStatsPass class."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class RecordStatsPass(BasePass):
    """
    The RecordStatsPass class.

    The RecordStatsPass stores stats about the circuit.
    """

    key = 'RecordStatsPass_stats_list'

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        stats: dict[str, Any] = {}
        stats['cycles'] = circuit.num_cycles
        stats['num_ops'] = circuit.num_operations
        stats['cgraph'] = circuit.coupling_graph
        stats['depth'] = circuit.depth
        stats['gate_counts'] = {
            gate: circuit.count(gate)
            for gate in circuit.gate_set
        }

        if self.key not in data:
            data[self.key] = []

        data[self.key].append(stats)
