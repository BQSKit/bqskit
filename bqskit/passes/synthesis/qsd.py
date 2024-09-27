"""This module implements the Quantum Shannon Decomposition."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.linalg import cossin
from scipy.linalg import schur

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitLocation
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant import CNOTGate
from bqskit.ir.gates.parameterized import RYGate
from bqskit.ir.gates.parameterized import RZGate
from bqskit.ir.gates.parameterized import VariableUnitaryGate
from bqskit.ir.gates.parameterized.mpry import MPRYGate
from bqskit.ir.gates.parameterized.mprz import MPRZGate
from bqskit.ir.operation import Operation
from bqskit.passes.processing import ScanningGateRemovalPass
from bqskit.passes.processing import TreeScanningGateRemovalPass
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class FullQSDPass(BasePass):
    """
    A pass performing one round of decomposition from the QSD algorithm.

    Important: This pass runs on VariableUnitaryGates only. Make sure to convert
    any gates you want to decompose to VariableUnitaryGates before running this
    pass.

    Additionally, ScanningGateRemovalPass will operate on the context of the
    entire circuit. If your circuit is large, it is best to set `perform_scan`
    to False.

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
        min_qudit_size: int = 2,
        perform_scan: bool = False,
        start_from_left: bool = True,
        tree_depth: int = 0,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a single level of the QSDPass.

        Args:
            min_qudit_size (int): Performs QSD until the circuit only has
                VariableUnitaryGates with a number of qudits less than or equal
                to this value. (Default: 2)
            perform_scan (bool): Whether or not to perform the scanning
                gate removal pass. (Default: False)
            start_from_left (bool): Determines where the scan starts
                attempting to remove gates from. If True, scan goes left
                to right, otherwise right to left. (Default: True)
            tree_depth (int): The depth of the tree to use in the
                TreeScanningGateRemovalPass. If set to 0, we will instead
                use the ScanningGateRemovalPass. (Default: 0)
            instantiate_options (dict): The options to pass to the
                scanning gate removal pass. (Default: {})
        """
        self.start_from_left = start_from_left
        self.min_qudit_size = min_qudit_size
        instantiation_options = {'method': 'qfactor'}
        instantiation_options.update(instantiate_options)
        self.scan = ScanningGateRemovalPass(
            start_from_left=start_from_left,
            instantiate_options=instantiation_options,
        )
        if tree_depth > 0:
            self.scan = TreeScanningGateRemovalPass(
                start_from_left=start_from_left,
                instantiate_options=instantiation_options,
                tree_depth=tree_depth,
            )
        # Instantiate the helper QSD pass
        self.qsd = QSDPass(min_qudit_size=min_qudit_size)
        # Instantiate the helper Multiplex Gate Decomposition pass
        self.mgd = MGDPass()
        self.perform_scan = perform_scan

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Run a round of QSD, Multiplex Gate Decomposition, and Scanning Gate
        Removal (optionally) until you reach the desired qudit size gates."""
        passes: list[BasePass] = []
        start_num = max(x.num_qudits for x in circuit.operations())
        for _ in range(self.min_qudit_size, start_num):
            passes.append(self.qsd)
            if self.perform_scan:
                passes.append(self.scan)
            passes.append(self.mgd)
        await Workflow(passes).run(circuit, data)


class MGDPass(BasePass):
    """
    A pass performing one round of decomposition of the MPRY and MPRZ gates in a
    circuit.

    References:
        C.C. Paige, M. Wei,
        History and generality of the CS decomposition,
        Linear Algebra and its Applications,
        Volumes 208–209,
        1994,
        Pages 303-326,
        ISSN 0024-3795,
        https://arxiv.org/pdf/quant-ph/0406176.pdf
    """

    @staticmethod
    def decompose(op: Operation) -> Circuit:
        """
        Return the decomposed circuit from one operation of a multiplexed gate.

        Args:
            op (Operation): The operation to decompose.

        Returns:
            Circuit: The decomposed circuit.
        """

        # Final level of decomposition decomposes to RY or RZ gate
        gate: MPRYGate | MPRZGate | RYGate | RZGate = MPRZGate(
            op.num_qudits - 1,
            op.num_qudits - 2,
        )
        if (op.num_qudits > 2):
            if isinstance(op.gate, MPRYGate):
                gate = MPRYGate(op.num_qudits - 1, op.num_qudits - 2)
        elif (isinstance(op.gate, MPRYGate)):
            gate = RYGate()
        else:
            gate = RZGate()

        left_params, right_params = MPRYGate.get_decomposition(op.params)

        # Construct Circuit
        circ = Circuit(op.gate.num_qudits)
        new_gate_location = list(range(1, op.gate.num_qudits))
        cx_location = (0, op.gate.num_qudits - 1)
        # print(type(gate), gate.num_qudits, new_gate_location)
        circ.append_gate(gate, new_gate_location, left_params)
        circ.append_gate(CNOTGate(), cx_location)
        circ.append_gate(gate, new_gate_location, right_params)
        circ.append_gate(CNOTGate(), cx_location)

        return circ

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Decompose all MPRY and MPRZ gates in the circuit one level."""
        gates = []
        pts = []
        locations = []
        num_ops = 0
        all_ops = list(circuit.operations_with_cycles(reverse=True))

        # Gather all of the multiplexed operations
        for cyc, op in all_ops:
            if isinstance(op.gate, MPRYGate) or isinstance(op.gate, MPRZGate):
                num_ops += 1
                gates.append(op)
                pts.append((cyc, op.location[0]))
                locations.append(op.location)

        if len(gates) > 0:
            # Do a bulk QSDs -> circs
            circs = [MGDPass.decompose(gate) for gate in gates]
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [
                Operation(x, locations[i], x._circuit.params)
                for i, x in enumerate(circ_gates)
            ]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        circuit.unfold_all()


def shift_down_unitary(num_qudits: int, end_qubits: int) -> PermutationMatrix:
    top_qubits = num_qudits - end_qubits
    now_bottom_qubits = list(reversed(range(top_qubits)))
    now_top_qubits = list(range(num_qudits - end_qubits, num_qudits))
    final_qudits = now_top_qubits + now_bottom_qubits
    return PermutationMatrix.from_qubit_location(num_qudits, final_qudits)


def shift_up_unitary(num_qudits: int, end_qubits: int) -> PermutationMatrix:
    bottom_qubits = list(range(end_qubits))
    top_qubits = list(reversed(range(end_qubits, num_qudits)))
    final_qudits = top_qubits + bottom_qubits
    return PermutationMatrix.from_qubit_location(num_qudits, final_qudits)


class QSDPass(BasePass):
    """
    A pass performing one round of decomposition from the QSD algorithm.

    This decomposition takes each unitary of size n and decomposes it
    into a circuit with 4 VariableUnitaryGates of size n - 1 and 3 multiplexed
    rotation gates.

    Important: This pass runs on VariableUnitaryGates only.

    References:
        https://arxiv.org/pdf/quant-ph/0406176
    """

    def __init__(
        self,
        min_qudit_size: int = 4,
    ) -> None:
        """
        Construct a single level of the QSDPass.

        Args:
            min_qudit_size (int): Performs a decomposition on all gates
                with width > min_qudit_size
        """
        self.min_qudit_size = min_qudit_size

    @staticmethod
    def create_unitary_gate(u: UnitaryMatrix) -> tuple[
        VariableUnitaryGate,
        RealVector,
    ]:
        """Create a VariableUnitaryGate from a UnitaryMatrix."""
        gate = VariableUnitaryGate(u.num_qudits)
        params = np.concatenate((np.real(u).flatten(), np.imag(u).flatten()))
        return gate, params

    @staticmethod
    def create_multiplexed_circ(
        us: list[UnitaryMatrix],
        select_qubits: list[int],
    ) -> Circuit:
        """
        Takes a list of 2 unitaries of size n. Returns a circuit that decomposes
        the unitaries into a circuit with 2 unitaries of size n-1 and a
        multiplexed controlled gate.

        Args:
            us (list[UnitaryMatrix]): The unitaries to decompose
            select_qubits (list[int]): The qubits to use as select qubits
            controlled_qubit (int): The qubit to use as the controlled qubit

        Returns:
            Circuit: The circuit that decomposes the unitaries

        Using this paper: https://arxiv.org/pdf/quant-ph/0406176.pdf. Thm 12
        """
        u1 = us[0]
        u2 = us[1]
        assert (u1.num_qudits == u2.num_qudits)
        all_qubits = list(range(len(select_qubits) + 1))
        # Use Schur Decomposition to split Us into V, D, and W matrices
        D_2, V = schur(u1._utry @ u2.dagger._utry)
        # D^2 will be diagonal since u1u2h is unitary
        D = np.sqrt(np.diag(np.diag(D_2)))
        # Calculate W @ U1
        left_mat = D @ V.conj().T @ u2._utry
        left_gate, left_params = QSDPass.create_unitary_gate(
            UnitaryMatrix(left_mat),
        )

        # Create Multi Controlled Z Gate
        z_params: RealVector = np.array(-2 * np.angle(np.diag(D)).flatten())
        z_gate = MPRZGate(len(all_qubits), u1.num_qudits)

        # Create right gate
        right_gate, right_params = QSDPass.create_unitary_gate(UnitaryMatrix(V))

        circ = Circuit(u1.num_qudits + 1)
        circ.append_gate(left_gate, CircuitLocation(select_qubits), left_params)
        circ.append_gate(z_gate, CircuitLocation(all_qubits), z_params)
        circ.append_gate(
            right_gate, CircuitLocation(
                select_qubits,
            ), right_params,
        )
        return circ

    @staticmethod
    def mod_unitaries(u: UnitaryMatrix) -> UnitaryMatrix:
        """Apply a permutation transform to the unitaries to the rest of the
        circuit."""
        shift_up = shift_up_unitary(u.num_qudits, u.num_qudits - 1)
        shift_down = shift_down_unitary(u.num_qudits, u.num_qudits - 1)
        return shift_up @ u @ shift_down

    @staticmethod
    def qsd(orig_u: UnitaryMatrix) -> Circuit:
        '''
        Perform the Quantum Shannon Decomposition on a unitary matrix.
        Args:
            orig_u (UnitaryMatrix): The unitary matrix to decompose

        Returns:
            Circuit: The circuit that decomposes the unitary
        '''

        # Shift the unitary qubits down by one
        u = QSDPass.mod_unitaries(orig_u)

        # Perform CS Decomp to solve for multiplexed unitaries and theta_y
        (u1, u2), theta_y, (v1h, v2h) = cossin(
            u._utry, p=u.shape[0] / 2, q=u.shape[1] / 2, separate=True,
        )
        assert (len(theta_y) == u.shape[0] / 2)

        # Create the multiplexed circuit
        # This generates 2 circuits that multipex U,V with an MPRY gate
        controlled_qubit = u.num_qudits - 1
        select_qubits = list(range(0, u.num_qudits - 1))
        all_qubits = list(range(u.num_qudits))
        circ_1 = QSDPass.create_multiplexed_circ(
            [
                UnitaryMatrix(v1h), UnitaryMatrix(v2h),
            ],
            select_qubits,
        )
        circ_2 = QSDPass.create_multiplexed_circ(
            [
                UnitaryMatrix(u1), UnitaryMatrix(u2),
            ],
            select_qubits,
        )
        gate_2 = MPRYGate(u.num_qudits, controlled_qubit)

        circ_1.append_gate(gate_2, CircuitLocation(all_qubits), 2 * theta_y)
        circ_1.append_circuit(
            circ_2, CircuitLocation(list(range(u.num_qudits))),
        )
        return circ_1

    async def run(self, circuit: Circuit, data: PassData) -> None:
        unitaries = []
        pts = []
        locations = []
        num_ops = 0
        all_ops = list(circuit.operations_with_cycles(reverse=True))

        initial_utry = circuit.get_unitary()
        # Gather all of the VariableUnitary unitaries
        for cyc, op in all_ops:
            if (
                op.num_qudits > self.min_qudit_size
                and isinstance(op.gate, VariableUnitaryGate)
            ):
                num_ops += 1
                unitaries.append(op.get_unitary())
                pts.append((cyc, op.location[0]))
                locations.append(op.location)

        if len(unitaries) > 0:
            circs = await get_runtime().map(QSDPass.qsd, unitaries)
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [
                Operation(x, locations[i], x._circuit.params)
                for i, x in enumerate(circ_gates)
            ]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        dist = circuit.get_unitary().get_distance_from(initial_utry)

        assert dist < 1e-5

        circuit.unfold_all()
