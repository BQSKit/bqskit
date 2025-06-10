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
from bqskit.ir.circuit import CircuitPoint
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
        # self.mgd = MGDPass(decompose_twice=False)
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

    def __init__(self, decompose_twice: bool = True) -> None:
        """
        The MGDPass decomposes all MPRY and MPRZ gates in a circuit.

        Args:
            decompose_twice (bool): Whether to decompose the MPRZ gate twice.
            This will save 2 CNOT gates in the decomposition. If false,
            the pass will only decompose one level. (Default: True)
        """
        self.decompose_twice = decompose_twice
        self.decompose_all = False

    def __init__(self, decompose_all: bool) -> None:
        """
        The MGDPass decomposes all MPRY and MPRZ gates in a circuit.

        Args:
            decompose_twice (bool): Whether to decompose the MPRZ gate twice.
            This will save 2 CNOT gates in the decomposition. If false,
            the pass will only decompose one level. (Default: True)
        """
        self.decompose_twice = not decompose_all
        self.decompose_all = decompose_all

    # alternate constructor
    def __init__(self, decompose_twice: bool, decompose_all: bool) -> None:
        """
        The MGDPass decomposes all MPRY and MPRZ gates in a circuit.

        Args:
            decompose_all (bool): Whether to decompose the MPRZ gate all the way.
            This will save 2^k CNOT gates in the decomposition. If false,
            the pass will only decompose two levels. (Default: True)
        """
        self.decompose_twice = decompose_twice
        self.decompose_all = decompose_all

    @staticmethod
    def decompose_mpx_all_levels(
        decompose_ry: bool, 
        params: RealVector, 
        num_qudits: int, 
        reverse: bool = False, 
        drop_last_cnot: bool = False
    ) -> Circuit:
        """
        Decompose a multiplexed RZ gate all levels 
        """

        if num_qudits < 4:
            # If you have less than 4 qubits, decompose two levels
            return MGDPass.decompose_mpx_two_levels(
                decompose_ry,
                params,
                num_qudits,
                reverse,
                drop_last_cnot=drop_last_cnot
            )

        # Decompose current level
        left_params, right_params = MPRYGate.get_decomposition(params)

        circ_left = MGDPass.decompose_mpx_all_levels(
            decompose_ry=decompose_ry,
            params=left_params,
            num_qudits=num_qudits - 1,
            reverse=reverse,
            drop_last_cnot=True
        )

        circ_right = MGDPass.decompose_mpx_all_levels(
            decompose_ry=decompose_ry,
            params=right_params,
            num_qudits=num_qudits - 1,
            reverse=not reverse,
            drop_last_cnot=True
        )

        circ = Circuit(num_qudits)
        cx_location_big = (0, num_qudits - 1)

        ops = [
            circ_left,
            Operation(CNOTGate(), cx_location_big),
            circ_right,
            Operation(CNOTGate(), cx_location_big),
        ]

        if drop_last_cnot:
            ops.pop()

        if reverse:
            ops.reverse()
        for op in ops:
            if isinstance(op, Operation):
                circ.append(op)
            else:
                circ.append_circuit(op, list(range(1, num_qudits)))

        # if drop_last_cnot:
        #     ops.pop()
        
        return circ

    

    @staticmethod
    def decompose_mpx_one_level(
        decompose_ry: bool,
        params: RealVector,
        num_qudits: int,
        reverse: bool = False,
        drop_last_cnot: bool = False,
    ) -> Circuit:
        """
        Decompose Multiplexed Gate one level.

        Args:
            params (RealVector): The parameters for the original MPRZ gate
            num_qudits (int): The number of qudits in the MPRZ gate
            reverse (bool): Whether to reverse the order of the gates (you can
            decompose the gates in either order to get the same result)
            drop_last_cnot (bool): Whether to drop the last CNOT gate. This
            should be set if you are doing a 2 level decomposition to save 2
            CNOT gates.

        Returns:
            Circuit: The circuit that decomposes the MPRZ gate
        """

        new_gate: MPRZGate | MPRYGate | RZGate | RYGate = RZGate()
        if decompose_ry:
            new_gate = RYGate()

        if (num_qudits >= 3):
            if decompose_ry:
                new_gate = MPRYGate(num_qudits - 1, num_qudits - 2)
            else:
                # Remove 1 qubit, last qubit is controlled
                new_gate = MPRZGate(num_qudits - 1, num_qudits - 2)

        left_params, right_params = MPRYGate.get_decomposition(params)
        circ = Circuit(num_qudits)
        new_gate_location = list(range(1, num_qudits))
        cx_location = (0, num_qudits - 1)

        ops = [
            Operation(new_gate, new_gate_location, left_params),
            Operation(CNOTGate(), cx_location),
            Operation(new_gate, new_gate_location, right_params),
            Operation(CNOTGate(), cx_location),
        ]

        if drop_last_cnot:
            ops.pop()

        if reverse:
            ops.reverse()

        for op in ops:
            circ.append(op)

        return circ

    @staticmethod
    def decompose_mpx_two_levels(
        decompose_ry: bool,
        params: RealVector,
        num_qudits: int,
        reverse: bool = False,
        drop_last_cnot: bool = False,
    ) -> Circuit:
        """
        We decompose a multiplexed RZ gate 2 levels deep. This allows you to
        remove 2 CNOTs as per Figure 2 in
        https://arxiv.org/pdf/quant-ph/0406176.pdf.

        Furthermore, in the context of the Block ZXZ decomposition, you can
        set `drop_last_cnot` to True. This CNOT gets merged into a central gate,
        which saves another 2 CNOTs. This is shown in section 5.2 of
        https://arxiv.org/pdf/2403.13692v1.pdf.

        Args:
            decompose_ry (bool): Whether to decompose the MPRY gate
            params (RealVector): The parameters for the original MPR gate
            num_qudits (int): The number of qudits in the MPR gate
            reverse (bool): Whether to reverse the order of the gates (you can
            decompose the gates in either order to get the same result)
            drop_last_cnot (bool): Whether to drop the last CNOT gate (only
            should be set to True if you are doing section 5.2 optimization)

        Returns:
            Circuit: The circuit that decomposes the MPR gate
        """

        if num_qudits <= 2:
            # If you have less than 3 qubits, just decompose one level
            return MGDPass.decompose_mpx_one_level(
                decompose_ry,
                params,
                num_qudits,
                reverse,
                drop_last_cnot=drop_last_cnot
            )

        # Get params for first decomposition of the MPRZ gate
        left_params, right_params = MPRYGate.get_decomposition(params)

        # Decompose the MPRZ gate into 2 MPRZ gates, dropping the last CNOT
        # Also Reverse the circuit for the right side in order to do the
        # optimization in section 5.2
        circ_left = MGDPass.decompose_mpx_one_level(
            decompose_ry,
            left_params,
            num_qudits - 1,
            reverse,
            drop_last_cnot=True,
        )

        circ_right = MGDPass.decompose_mpx_one_level(
            decompose_ry,
            right_params,
            num_qudits - 1,
            not reverse,
            drop_last_cnot=True,
        )

        # Now, construct the circuit.
        # This will generate the original MPRZ gate with the target qubit
        # set as qubit num_qudits - 1
        circ = Circuit(num_qudits)
        cx_location_big = (0, num_qudits - 1)

        ops: list[Circuit | Operation] = [
            circ_left,
            Operation(CNOTGate(), cx_location_big),
            circ_right,
            Operation(CNOTGate(), cx_location_big),
        ]

        # We can remove the last CNOT gate as per section 5.2
        if drop_last_cnot:
            ops.pop()

        if reverse:
            ops.reverse()

        for op in ops:
            if isinstance(op, Operation):
                circ.append(op)
            else:
                circ.append_circuit(op, list(range(1, num_qudits)))

        return circ

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Decompose all MPRY and MPRZ gates in the circuit one level."""
        ops: list[Operation] = []
        pts: list[CircuitPoint] = []
        locations: list[CircuitLocation] = []
        all_ops = list(circuit.operations_with_cycles(reverse=True))

        # Gather all of the multiplexed operations
        for cyc, op in all_ops:
            if isinstance(op.gate, MPRYGate) or isinstance(op.gate, MPRZGate):
                ops.append(op)
                pts.append(CircuitPoint((cyc, op.location[0])))
                # Adjust location based on current target, move target to last
                # qudit
                loc = list(op.location)
                loc = (
                    loc[0:op.gate.target_qubit]
                    + loc[(op.gate.target_qubit + 1):]
                    + [loc[op.gate.target_qubit]]
                )
                locations.append(CircuitLocation(loc))

        if len(ops) > 0:
            # Do a bulk QSDs -> circs
            if self.decompose_all:
                circs = [
                    MGDPass.decompose_mpx_all_levels(
                        isinstance(op.gate, MPRYGate),
                        op.params,
                        op.num_qudits,
                    ) for op in ops
                ]

            elif self.decompose_twice:
                circs = [
                    MGDPass.decompose_mpx_two_levels(
                        isinstance(op.gate, MPRYGate),
                        op.params,
                        op.num_qudits,
                    ) for op in ops
                ]

            else:
                circs = [
                    MGDPass.decompose_mpx_one_level(
                        isinstance(op.gate, MPRYGate),
                        op.params,
                        op.num_qudits,
                    ) for op in ops
                ]

            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [
                Operation(x, locations[i], x._circuit.params)
                for i, x in enumerate(circ_gates)
            ]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        circuit.unfold_all()


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
    def shift_down_unitary(
        num_qudits: int,
        end_qubits: int,
    ) -> PermutationMatrix:
        """Return the Permutation Matrix that shifts the qubits down by 1
        qubit."""
        top_qubits = num_qudits - end_qubits
        now_bottom_qubits = list(reversed(range(top_qubits)))
        now_top_qubits = list(range(num_qudits - end_qubits, num_qudits))
        final_qudits = now_top_qubits + now_bottom_qubits
        return PermutationMatrix.from_qubit_location(num_qudits, final_qudits)

    @staticmethod
    def shift_up_unitary(
        num_qudits: int,
        end_qubits: int,
    ) -> PermutationMatrix:
        """Return the Permutation Matrix that shifts the qubits down by 1
        qubit."""
        bottom_qubits = list(range(end_qubits))
        top_qubits = list(reversed(range(end_qubits, num_qudits)))
        final_qudits = top_qubits + bottom_qubits
        return PermutationMatrix.from_qubit_location(num_qudits, final_qudits)

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
        shift_up = QSDPass.shift_up_unitary(u.num_qudits, u.num_qudits - 1)
        shift_down = QSDPass.shift_down_unitary(u.num_qudits, u.num_qudits - 1)
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

    @staticmethod
    def get_variable_unitary_pts(
        circuit: Circuit,
        min_qudit_size: int,
    ) -> tuple[list[UnitaryMatrix], list[CircuitPoint], list[CircuitLocation]]:
        """Get all VariableUnitary Gates in the circuit wider than
        `min_qudit_size` and return their unitaries, points, and locations."""
        unitaries: list[UnitaryMatrix] = []
        pts: list[CircuitPoint] = []
        locations: list[CircuitLocation] = []
        num_ops = 0
        all_ops = list(circuit.operations_with_cycles(reverse=True))

        # Gather all of the VariableUnitary unitaries
        for cyc, op in all_ops:
            if (
                op.num_qudits > min_qudit_size
                and isinstance(op.gate, VariableUnitaryGate)
            ):
                num_ops += 1
                unitaries.append(op.get_unitary())
                pts.append(CircuitPoint((cyc, op.location[0])))
                locations.append(op.location)

        return unitaries, pts, locations

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform a single pass of Quantum Shannon Decomposition on the
        circuit."""
        unitaries, pts, locations = QSDPass.get_variable_unitary_pts(
            circuit, self.min_qudit_size,
        )

        if len(unitaries) > 0:
            circs = await get_runtime().map(QSDPass.qsd, unitaries)
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [
                Operation(x, locations[i], x._circuit.params)
                for i, x in enumerate(circ_gates)
            ]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        circuit.unfold_all()
