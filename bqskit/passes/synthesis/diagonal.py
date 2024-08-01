"""This module implements the WalshDiagonalSynthesisPass."""
from __future__ import annotations

import logging

from numpy import where

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RZGate
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.math import pauliz_expansion
from bqskit.utils.math import unitary_log_no_i


_logger = logging.getLogger(__name__)


class WalshDiagonalSynthesisPass(SynthesisPass):
    """
    A pass that synthesizes diagonal unitaries into Walsh functions.

    Based on: https://arxiv.org/abs/1306.3991
    """

    def __init__(
        self,
        parameter_precision: float = 1e-8,
    ) -> None:
        """
        Constructor for WalshDiagonalSynthesisPass.

        Args:
            parameter_precision (float): Pauli strings with parameter values
                less than this are rounded to zero. (Default: 1e-8)

        TODO:
            - Cancel adjacent CNOTs
            - See how QFAST can be used to generalize to qudits
        """
        self.parameter_precision = parameter_precision

    def gray_code(self, number: int) -> int:
        """Convert a number to its Gray code representation."""
        gray = number ^ (number >> 1)
        return gray

    def pauli_to_subcircuit(
        self,
        string_id: int,
        angle: float,
        num_qubits: int,
    ) -> Circuit:
        string = bin(string_id)[2:].zfill(num_qubits)
        circuit = Circuit(num_qubits)
        locations = [i for i in range(num_qubits) if string[i] == '1']
        if len(locations) == 1:
            circuit.append_gate(RZGate(), locations[0], [angle])
        elif len(locations) > 1:
            pairs = [
                (locations[i], locations[i + 1])
                for i in range(len(locations) - 1)
            ]
            for pair in pairs:
                circuit.append_gate(CNOTGate(), pair)
            circuit.append_gate(RZGate(), locations[-1], [angle])
            for pair in reversed(pairs):
                circuit.append_gate(CNOTGate(), pair)
        return circuit

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        if not isinstance(utry, UnitaryMatrix):
            m = 'WalshDiagonalSynthesisPass can only synthesize diagonal, '
            m += f'`UnitaryMatrix`s, got {type(utry)}.'
            raise TypeError(m)

        if not utry.is_qubit_only():
            m = 'WalshDiagonalSynthesisPass can only synthesize diagonal '
            m += '`UnitaryMatrix`s with qubits, got higher radix than 2.'
            raise ValueError(m)

        num_qubits = utry.num_qudits
        circuit = Circuit(num_qubits)

        # Find parameters of each I/Z Pauli string
        H_matrix = unitary_log_no_i(utry.numpy)
        params = pauliz_expansion(H_matrix) * 2
        # Remove low weight terms - these are likely numerical errors
        params = where(abs(params) < self.parameter_precision, 0, params)

        # Order the Pauli strings by their Gray code representation
        pauli_params = sorted(
            [(i, -p) for i, p in enumerate(params)],
            key=lambda x: self.gray_code(x[0]),
        )
        subcircuits = [
            self.pauli_to_subcircuit(i, p, num_qubits) for i, p in pauli_params
        ]

        for subcircuit in subcircuits:
            circuit.append_circuit(subcircuit, [_ for _ in range(num_qubits)])

        return circuit
