"""This module implements the Rebase2QuditGatePass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir import Gate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import U3Gate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.point import CircuitPoint as Point
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from bqskit.utils.typing import is_sequence
_logger = logging.getLogger(__name__)


class Rebase2QuditGatePass(BasePass):
    """
    The Rebase2QuditGatePass class.

    Will use instantiation to change the a 2-qudit gate to a different one.
    """

    def __init__(
        self,
        gate_in_circuit: Gate | Sequence[Gate],
        new_gate: Gate | Sequence[Gate],
        max_depth: int = 3,
        max_retries: int = -1,
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a Rebase2QuditGatePass.

        Args:
            gate_in_circuit (Gate | Sequence[Gate]): The two-qudit gate
                or gates in the circuit that you want to replace.

            new_gate (Gate | Sequence[Gate]): The two-qudit new gate or
                gates you want to put in the circuit.

            max_depth (int): The maximum number of new gates to replace
                an old gate with. (Default: 3)

            max_retries (int): The number of retries for the same gate
                before we increase the maximum depth. If left as -1,
                then never increase max depth. (Default: -1)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `gate_in_circuit` or `new_gate` is not a 2-qudit
                gate.

            ValueError: if `max_depth` is nonnegative.
        """

        if is_sequence(gate_in_circuit):
            if any(not isinstance(g, Gate) for g in gate_in_circuit):
                raise TypeError('Expected Gate or Gate list.')

        elif not isinstance(gate_in_circuit, Gate):
            raise TypeError(f'Expected Gate, got {type(gate_in_circuit)}.')

        else:
            gate_in_circuit = [gate_in_circuit]

        if any(g.num_qudits != 2 for g in gate_in_circuit):
            raise ValueError('Expected 2-qudit gate.')

        if is_sequence(new_gate):
            if any(not isinstance(g, Gate) for g in new_gate):
                raise TypeError('Expected Gate or Gate list.')

        elif not isinstance(new_gate, Gate):
            raise TypeError(f'Expected Gate, got {type(new_gate)}.')

        else:
            new_gate = [new_gate]

        if any(g.num_qudits != 2 for g in new_gate):
            raise ValueError('Expected 2-qudit gate.')

        if not is_integer(max_depth):
            raise TypeError(f'Expected integer, got {max_depth}.')

        if max_depth < 0:
            raise ValueError(f'Expected nonnegative depth, got: {max_depth}.')

        if not is_integer(max_retries):
            raise TypeError(f'Expected integer, got {max_retries}.')

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.gates = gate_in_circuit
        self.ngates = new_gate
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
        }
        self.instantiate_options.update(instantiate_options)
        self.generate_new_gate_templates()

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug(f'Rebasing gates from {self.gates} to {self.ngates}.')

        target = self.get_target(circuit, data)

        for g in self.gates:
            # Track retries to check for no progress
            num_retries = 0
            prev_count = circuit.count(g)

            while g in circuit.gate_set:
                # Check if we made progress from last loop
                gates_left = circuit.count(g)
                if prev_count == gates_left:
                    num_retries += 1
                else:
                    prev_count = gates_left
                    num_retries = 0

                # Group together a 2-qubit block composed of gates from old set
                point = self.group_near_gates(circuit, circuit.point(g))
                circuits_with_new_gate = []
                for circ in self.circs:
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, circ)
                    circuits_with_new_gate.append(circuit_copy)

                # If we have exceeded the number of retries, up the max depth
                if self.max_retries >= 0 and num_retries > self.max_retries:
                    _logger.info('Exceeded max retries, increasing depth.')
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, self.overdrive)
                    circuits_with_new_gate.append(circuit_copy)

                instantiated_circuits = self.execute(
                    data,
                    Circuit.instantiate,
                    circuits_with_new_gate,
                    target=target,
                    **self.instantiate_options,
                )

                dists = [self.cost(c, target) for c in instantiated_circuits]

                # Find the successful circuit with the least gates
                best_index = None
                best_count = self.max_depth + 2
                for i, dist in enumerate(dists):
                    if dist < self.success_threshold:
                        if self.counts[i] < best_count:
                            best_index = i
                            best_count = self.counts[i]

                if best_index is None:
                    circuit.unfold(point)
                    continue

                _logger.info(self.replaced_log_messages[best_index])
                circuit.become(instantiated_circuits[best_index])

    def group_near_gates(self, circuit: Circuit, center: Point) -> Point:
        """Group gates similar to the gate at center on the same qubits."""
        op = circuit[center]
        qubits = op.location
        counts = {g: 0.0 for g in self.gates}
        counts[op.gate] += 1.0

        # Go to the left until cant
        i = 0
        moving_left = True
        while moving_left:
            i += 1
            if center.cycle - i < 0:
                i = center.cycle
                break
            for q in qubits:
                point = (center.cycle - i, q)
                if not circuit.is_point_idle(point):
                    lop = circuit[point]
                    if any(p not in qubits for p in lop.location):
                        i -= 1
                        moving_left = False
                        break
                    if lop.num_qudits != 1 and lop.gate not in self.gates:
                        i -= 1
                        moving_left = False
                        break
                    if lop.num_qudits == 2:
                        counts[lop.gate] += 0.5

        j = 0
        moving_right = True
        while moving_right:
            j += 1
            if center.cycle + j >= circuit.num_cycles:
                j = circuit.num_cycles - center.cycle - 1
                break
            for q in qubits:
                point = (center.cycle + j, q)
                if not circuit.is_point_idle(point):
                    rop = circuit[point]
                    if any(p not in qubits for p in rop.location):
                        j -= 1
                        moving_right = False
                        break
                    if rop.num_qudits != 1 and rop.gate not in self.gates:
                        j -= 1
                        moving_right = False
                        break
                    if rop.num_qudits == 2:
                        counts[rop.gate] += 0.5

        region = {q: (center.cycle - i, center.cycle + j) for q in qubits}
        grouped_gate_str = ', '.join([
            f'{int(c)} {g}' + ('s' if c > 1 else '')
            for g, c in counts.items()
        ])
        _logger.info(f'Grouped together {grouped_gate_str}.')
        return circuit.fold(region)

    def generate_new_gate_templates(self) -> None:
        """Generate the templates to be instantiated over old circuits."""
        self.circs = []
        self.counts = []

        circ = Circuit(2)
        circ.append_gate(U3Gate(), 0)
        circ.append_gate(U3Gate(), 1)
        self.circs.append(circ)
        self.counts.append(0)

        for g in self.ngates:
            for i in range(1, self.max_depth + 1):
                circ = Circuit(2)
                circ.append_gate(U3Gate(), 0)
                circ.append_gate(U3Gate(), 1)

                for _ in range(i):
                    circ.append_gate(g, (0, 1))
                    circ.append_gate(U3Gate(), 0)
                    circ.append_gate(U3Gate(), 1)

                self.counts.append(i)
                self.circs.append(circ)

        # Add overdrive circuit, incase we exceed retry limit
        self.overdrive = self.circs[-1].copy()
        self.overdrive.append_gate(self.ngates[-1], (0, 1))
        self.overdrive.append_gate(U3Gate(), 0)
        self.overdrive.append_gate(U3Gate(), 1)
        self.counts.append(self.counts[-1] + 1)

        # Preprocess log messages
        self.replaced_log_messages = []
        for circ in self.circs + [self.overdrive]:
            counts = [circ.count(g) for g in circ.gate_set]
            gate_count_str = ', '.join([
                f'{c} {g}' + ('s' if c > 1 else '')
                for c, g in zip(counts, circ.gate_set)
            ])
            msg = f'Replaced gate with {gate_count_str}.'
            self.replaced_log_messages.append(msg)
