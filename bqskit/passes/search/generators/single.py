"""This module implements the SingleQuditLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Sequence

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_sequence
_logger = logging.getLogger(__name__)


class SingleQuditLayerGenerator(LayerGenerator):
    """
    The SingleQuditLayerGenerator class.

    This module implements a layer generator for a single-qudit circuits with
    hetergenous gate support.
    """

    def __init__(
        self,
        gates: Sequence[Gate] | None,
        allow_repeats: bool = False,
    ) -> None:
        """
        Construct a SingleQuditLayerGenerator.

        Args:
            gates (Sequence[Gate] | None): The single-qudit gate list. If None,
                then the generator will read the pass data for the model's
                gate set.

            allow_repeats (bool): By default this is set to False,
                and the generator will not generate a circuit that
                has the same gate twice in a row. Some gates are not
                closed under multiplication and it may be beneficial
                to allow them to repeat. In this case, `allow_repeats`
                should be True.

        Raises:
            ValueError: If any pair of gates from `gates` has a
                different radix.

            ValueError: If any gate from `gates` is not single-qudit.

            ValueError: If `gates` is empty.
        """
        if gates is not None:
            self.radix = None
            if not is_sequence(gates):
                raise TypeError(
                    f'Expected sequence of gates, got {type(gates)}.',
                )

            for gate in gates:
                if not isinstance(gate, Gate):
                    raise TypeError(f'Expected gate, got {type(gate)}.')

                if gate.num_qudits > 1:
                    raise ValueError(f'Expected single-qudit gate, got {gate}.')

                if self.radix is None:
                    self.radix = gate.radixes[0]

                elif gate.radixes[0] != self.radix:
                    raise ValueError('Gate set has a radix inconsistency.')

            if len(gates) == 0:
                raise ValueError('Empty gate set.')

        self.gates = gates
        self.allow_repeats = allow_repeats

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` has a radix mismatch with
                `self.gates`.

            ValueError: If `target` is not single-qudit.
        """
        if self.gates is None:
            self.gates = list(data.gate_set.single_qudit_gates)
            self.radix = self.gates[0].radixes[0]
            if not all(g.radixes[0] == self.radix for g in self.gates):
                raise RuntimeError('Gate set has a radix inconsistency.')

        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if target.num_qudits > 1:
            raise ValueError('Target is larger than a single-qudit.')

        if target.radixes[0] != self.radix:
            raise ValueError('Mismatch between generator and target radix.')

        init_circuit = Circuit(target.num_qudits, target.radixes)
        return init_circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is not a single-qudit circuit.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits > 1:
            raise ValueError('Cannot expand a multi-qudit circuit.')

        # Record last gate
        last_gate = circuit[-1, -1].gate if len(circuit) >= 1 else None

        # Generate successors
        successors = []
        for gate in self.gates:  # type: ignore  # self.gates is set earlier

            # Skip adding a gate that repeats the last one on the circuit
            if not self.allow_repeats and gate == last_gate:
                continue

            successor = circuit.copy()
            successor.append_gate(gate, 0)
            successors.append(successor)

        return successors
