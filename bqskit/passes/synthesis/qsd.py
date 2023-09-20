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
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.gates import CircuitGate
from bqskit.runtime import get_runtime
from scipy.linalg import cossin, diagsvd, schur
import numpy as np
import time
from bqskit.qis.permutation import PermutationMatrix

_logger = logging.getLogger(__name__)

def shift_down_unitary(num_qudits: int, end_qubits: int):
    top_qubits = num_qudits - end_qubits
    now_bottom_qubits = list(reversed(range(top_qubits)))
    now_top_qubits = list(range(num_qudits - end_qubits, num_qudits))
    final_qudits = now_top_qubits + now_bottom_qubits
    return PermutationMatrix.from_qubit_location(num_qudits, final_qudits)

def shift_up_unitary(num_qudits: int, end_qubits: int):
    bottom_qubits = list(range(end_qubits))
    top_qubits = list(reversed(range(end_qubits, num_qudits)))
    final_qudits = top_qubits + bottom_qubits
    return PermutationMatrix.from_qubit_location(num_qudits, final_qudits)



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

    cs_time = 0
    schur_time = 0
    create_circ_time = 0
    append_circ_time = 0
    init_time = 0
    replace_time = 0

    def __init__(
            self,
            min_qudit_size: int = 4,
        ) -> None:
            """
            Construct a single level of the QSDPass.

            Args:
                min_qudit_size (int): Performs a decomposition on all gates
                    with widht > min_qudit_size
            """
            self.min_qudit_size = min_qudit_size

    @staticmethod
    def create_unitary_gate(u: UnitaryMatrix):
        gate = VariableUnitaryGate(u.num_qudits)
        params = np.concatenate((np.real(u).flatten(), np.imag(u).flatten()))
        # params = []
        return gate, params

    @staticmethod
    def create_multiplexed_circ(us: list[UnitaryMatrix], select_qubits: list[int], controlled_qubit: int) -> Circuit:
        '''Using this paper: https://arxiv.org/pdf/quant-ph/0406176.pdf. Thm 12'''
        # TODO: Expand to multiple unitaries
        u1 = us[0]
        u2 = us[1]
        assert(u1.num_qudits == u2.num_qudits)
        all_qubits = list(range(len(select_qubits) + 1))
        # First apply u1 gate
        # Now create controlled unitary of u1h @ u2
        # This breaks down into a Variable Unitary, Rz, and Variable unitary
        # # TODO: Eval Schur vs Eig
        start = time.time()
        D_2, V  = schur(u1._utry @ u2.dagger._utry)
        QSDPass.schur_time += (time.time() - start)
        D = np.sqrt(np.diag(np.diag(D_2))) # D^2 will be diagonal since u1u2h is unitary
        # Calculate W @ U1
        left_mat = D @ V.conj().T @ u2._utry
        left_gate, left_params = QSDPass.create_unitary_gate(UnitaryMatrix(left_mat))

        # Create Multi Controlled Z Gate
        z_params = 2 * np.angle(np.diag(D)).flatten()
        z_gate = MCRZGate(len(all_qubits), u1.num_qudits)

        # Create right gate
        right_gate, right_params = QSDPass.create_unitary_gate(UnitaryMatrix(V))

        circ = Circuit(u1.num_qudits + 1)
        circ.append_gate(left_gate, CircuitLocation(select_qubits), left_params)
        circ.append_gate(z_gate, CircuitLocation(all_qubits), z_params)
        circ.append_gate(right_gate, CircuitLocation(select_qubits), right_params)

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
    
    @staticmethod
    def mod_unitaries(u: UnitaryMatrix) -> UnitaryMatrix:
        shift_up = shift_up_unitary(u.num_qudits, u.num_qudits - 1)
        shift_down = shift_down_unitary(u.num_qudits, u.num_qudits - 1)
        return shift_up @ u @ shift_down

    @staticmethod
    def qsd(u: UnitaryMatrix) -> Circuit:
        '''
        Return the circuit that is generated from one levl of QSD. 
        '''
        start = time.time()
        (u1, u2), theta_y, (v1h, v2h) = cossin(u._utry, p=u.shape[0]/2, q=u.shape[1]/2, separate=True)
        QSDPass.cs_time += (time.time() - start)
        # print(QSDPass.cs_time)
        assert(len(theta_y) == u.shape[0] / 2)
        controlled_qubit = u.num_qudits - 1
        select_qubits = list(range(0, u.num_qudits - 1))
        all_qubits = list(range(u.num_qudits))
        start = time.time()
        circ_1 = QSDPass.create_multiplexed_circ([UnitaryMatrix(v1h), UnitaryMatrix(v2h)], select_qubits, controlled_qubit)
        circ_2 = QSDPass.create_multiplexed_circ([UnitaryMatrix(u1), UnitaryMatrix(u2)], select_qubits, controlled_qubit)
        QSDPass.create_circ_time += (time.time() - start)
        gate_2 = MCRYGate(u.num_qudits, controlled_qubit)
        start = time.time()
        circ_1.append_gate(gate_2, CircuitLocation(all_qubits), 2 * theta_y)
        circ_1.append_circuit(circ_2, CircuitLocation(list(range(u.num_qudits))))
        QSDPass.append_circ_time += (time.time() - start)
        return circ_1

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # while num_ops > 0:
        start = time.time()
        unitaries = []
        pts = []
        locations = []
        num_ops = 0
        all_ops = list(circuit.operations_with_cycles(reverse=True))
        # Gather all of the unitaries
        for cyc, op in all_ops:
            if op.num_qudits > self.min_qudit_size and not (isinstance(op.gate, MCRYGate) or isinstance(op.gate, MCRZGate)):
                num_ops += 1
                unitaries.append(op.get_unitary())
                pts.append((cyc, op.location[0]))
                locations.append(op.location)
        
        QSDPass.init_time += (time.time() - start)

        start = time.time()
        if len(unitaries) > 0:
            # Do a bulk QSDs -> circs
            unitaries = await get_runtime().map(QSDPass.mod_unitaries, unitaries)
            circs = await get_runtime().map(QSDPass.qsd, unitaries)
            # Do bulk replace (single threaded)
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [Operation(x, locations[i], x._circuit.params) for i,x in enumerate(circ_gates)]
            circuit.batch_replace(pts, circ_ops)
            # circuit.replace_with_circuit(pts[0], circ_ops[0])
            circuit.unfold_all()

        QSDPass.replace_time += (time.time() - start)

        # print(f"Init Time: {QSDPass.init_time}")
        # print(f"CS Time: {QSDPass.cs_time}")
        # print(f"Schur Time: {QSDPass.schur_time}")
        # print(f"Create Circ Time: {QSDPass.create_circ_time}")
        # print(f"Append Circ Time: {QSDPass.append_circ_time}")
        # print(f"Replace Time: {QSDPass.replace_time}")

        circuit.unfold_all()

        _logger.debug(f"Running Scanning gate removal on circuit with {circuit.gate_counts}")