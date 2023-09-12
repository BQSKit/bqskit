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
from bqskit.ir.gates.parameterized.cun import CUNGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.location import CircuitLocation
from scipy.linalg import cossin
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

    def create_multiplexed_circ(self, u1: UnitaryMatrix, u2: UnitaryMatrix, select_qubits: list[int]) -> Circuit:
        assert(u1.num_qudits == u2.num_qudits)
        # First apply u1 gate
        u1_gate = VariableUnitaryGate(u1.num_qudits)
        u1_params = np.real(u1._utry) + np.imag(u1._utry)
        # Now create controlled unitary of u1h @ u2
        inv_mat = u1.dagger()._utry @ u2._utry
        inv_params = np.real(inv_mat).flatten() + np.imag(inv_mat).flatten()
        inv_gate = CUNGate(u2.num_qudits + len(select_qubits), len(select_qubits))

        circ = Circuit(u1.num_qudits + len(select_qubits))
        controlled_bits = sorted(set(range(u1.num_qudits)).difference(set(select_qubits)))
        circ.append_gate(u1_gate, CircuitLocation(controlled_bits), u1_params)
        circ.append_gate(inv_gate, CircuitLocation(select_qubits + controlled_bits), inv_params)
        return circ

    async def qsd(self, u: UnitaryMatrix) -> Circuit:
        '''
        Return the circuit that is generated from one levl of QSD. 
        '''
        (u1, u2), theta_y, (v1h, v2h) = cossin(self._utry, p=self.shape[0]/2, q=self.shape[1]/2, separate=True)
        select_qubits = [0]
        controlled_qubits = list(range(1, u.num_qudits))
        circ_1 = self.create_multiplexed_circ(UnitaryMatrix(u1), UnitaryMatrix(u2), select_qubits)
        gate_2 = MCRYGate(u.num_qudits - 1)
        circ_1.append_gate(gate_2, CircuitLocation(select_qubits + controlled_qubits), theta_y)
        circ_2 = self.create_multiplexed_circ(UnitaryMatrix(v1h), UnitaryMatrix(v2h), select_qubits)
        circ_1.append_circuit(circ_2, CircuitLocation(list(range(u.num_qudits))))
        return circ_1



    async def perform_decomposition(self, circuit: Circuit, op: Operation, cycle: int) -> None:
        pt = circuit.point(op, (cycle, op.location[0]))
        new_circ = self.qsd(op.get_unitary())
        circuit.replace_with_circuit(pt, new_circ)
        return


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        
        for cyc, op in circuit.operations_with_cycles():
            if op.num_qudits > self.min_qudit_size:
                 self.perform_decomposition(circuit, op, cyc)