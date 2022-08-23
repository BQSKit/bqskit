"""This module implements the ExtractMeasurements and RestoreMeasurements
passes."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.point import CircuitPoint
_logger = logging.getLogger(__name__)


class ExtractMeasurements(BasePass):
    """A pass that stores and removes the measurements in a circuit."""

    key = '__measurement_data__'

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.info('Extracting measurements from circuit.')
        points_to_remove: list[CircuitPoint] = []
        cregs: dict[str, int] = {}
        measurements: dict[int, tuple[str, int]] = {}
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, MeasurementPlaceholder):
                points_to_remove.append(CircuitPoint(cycle, op.location[0]))
                measurements.update(op.gate.measurements)
                for name, size in op.gate.classical_regs:
                    if name not in cregs:
                        cregs[name] = size

        if len(cregs) > 0:
            circuit.batch_pop(points_to_remove)
            data[self.key] = (cregs, measurements)
            _logger.debug(f'Extracted classical registers: {cregs}.')
            _logger.debug(f'Extracted measurements: {measurements}.')


class RestoreMeasurements(BasePass):
    """A pass that restores the measurements in a circuit."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.info('Restoring measurements to circuit.')
        if ExtractMeasurements.key in data:
            cregs, measurements = data[ExtractMeasurements.key]

            # permute for final layout
            if 'final_layout' in data:
                pi = data['final_layout']
                measurements = {pi[q]: c for q, c in measurements.items()}

            mph = MeasurementPlaceholder(list(cregs.items()), measurements)
            circuit.append_gate(mph, list(measurements.keys()))
