"""This module implements the Quantum Shannon Decomposition for one level."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.operation import Operation
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.mcry import MCRYGate
from bqskit.ir.gates.parameterized.mcrz import MCRZGate
from bqskit.ir.gates.parameterized.cun import CUNGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.location import CircuitLocation
from scipy.linalg import cossin, diagsvd, schur
import numpy as np

_logger = logging.getLogger(__name__)


class QSDPass(BasePass):
    """
    A pass performing one round of decomposition from the QSD algorithm.

    References:
        C.C. Paige, M. Wei,
        History and generality of the CS decomposition,
        Linear Algebra and its Applications,
        Volumes 208–209,
        1994,
        Pages 303-326,
        ISSN 0024-3795,
        https://doi.org/10.1016/0024-3795(94)90446-4.
    """

    def __init__(
            self,
            start_from_left: bool = True,
            min_qudit_size: int = 4,
        ) -> None:
            """
            Construct a single level of the QSDPass.

            Args:
                start_from_left (bool): Determines where the scan starts
                    attempting to remove gates from. If True, scan goes left
                    to right, otherwise right to left. (Default: True)

                min_qudit_size (int): Performs a decomposition on all gates
                    with widht > min_qudit_size
            """

            self.start_from_left = start_from_left
            self.min_qudit_size = min_qudit_size

    def create_unitary_gate(self, u: UnitaryMatrix):
         gate = VariableUnitaryGate(u.num_qudits)
         params = np.concatenate((np.real(u).flatten(), np.imag(u).flatten()))
         return gate, params

    def create_multiplexed_circ(self, us: list[UnitaryMatrix], select_qubits: list[int], controlled_qubits: list[int]) -> Circuit:
        '''Using this paper: https://arxiv.org/pdf/quant-ph/0406176.pdf. Thm 12'''
        # TODO: Expand to multiple unitaries
        u1 = us[0]
        u2 = us[1]
        assert(2 ** len(select_qubits) == len(us))
        assert(u1.num_qudits == u2.num_qudits)
        # First apply u1 gate
        # Now create controlled unitary of u1h @ u2
        # This breaks down into a Variable Unitary, Rz, and Variable unitary
        D_2, V  = schur(u1._utry @ u2.dagger._utry)
        D = np.sqrt(np.diag(np.diag(D_2))) # D^2 will be diagonal since u1u2h is unitary
        # Calculate W @ U1
        left_mat = D @ V.conj().T @ u2._utry
        left_gate, left_params = self.create_unitary_gate(UnitaryMatrix(left_mat))

        # Create Multi Controlled Z Gate
        z_params = 2 * np.angle(np.diag(D)).flatten()
        z_gate = MCRZGate(len(select_qubits) + len(controlled_qubits), 0)

        # Create right gate
        right_gate, right_params = self.create_unitary_gate(UnitaryMatrix(V))

        circ = Circuit(u1.num_qudits + len(select_qubits))
        circ.append_gate(left_gate, CircuitLocation(controlled_qubits), left_params)
        circ.append_gate(z_gate, CircuitLocation(select_qubits + controlled_qubits), z_params)
        circ.append_gate(right_gate, CircuitLocation(controlled_qubits), right_params)

        # # Comparison!!!
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # orig_mat = np.identity(2 ** 3)
        # orig_mat[:4,:4] = u1._utry
        # orig_mat[4:, 4:] = u2._utry
        # print(orig_mat)

        # # New matrix from calculation
        # calc_mat = np.identity(2 ** 3)
        # calc_mat[:4, :4] = V @ D @ D @ V.conj().T @ u2._utry
        # calc_mat[4:, 4:] = V @ D.conj().T @ D @ V.conj().T @ u2._utry

        # print(calc_mat)
        # print(np.allclose)

        return circ

    def qsd(self, u: UnitaryMatrix) -> Circuit:
        '''
        Return the circuit that is generated from one levl of QSD. 
        '''
        (u1, u2), theta_y, (v1h, v2h) = cossin(u._utry, p=u.shape[0]/2, q=u.shape[1]/2, separate=True)
        assert(len(theta_y) == u.shape[0] / 2)
        select_qubits = [0]
        controlled_qubits = list(range(1, u.num_qudits))
        circ_1 = self.create_multiplexed_circ([UnitaryMatrix(v1h), UnitaryMatrix(v2h)], select_qubits, controlled_qubits)
        gate_2 = MCRYGate(u.num_qudits, 0)
        circ_1.append_gate(gate_2, CircuitLocation(select_qubits + controlled_qubits), 2 * theta_y)
        circ_2 = self.create_multiplexed_circ([UnitaryMatrix(u1), UnitaryMatrix(u2)], select_qubits, controlled_qubits)
        circ_1.append_circuit(circ_2, CircuitLocation(list(range(u.num_qudits))))
        return circ_1

    def perform_decomposition(self, circuit: Circuit, op: Operation, cycle: int) -> None:
        pt = circuit.point(op, (cycle, op.location[0]))
        new_circ = self.qsd(op.get_unitary())
        circuit.replace_with_circuit(pt, new_circ)
        return


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        
        for cyc, op in circuit.operations_with_cycles():
            if op.num_qudits > self.min_qudit_size:
                 self.perform_decomposition(circuit, op, cyc)