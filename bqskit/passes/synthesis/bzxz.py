"""This module implements the Full Block ZXZ Decomposition. Additionally, it
defines the Block ZXZ Decomposition for a single pass. """
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass

from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.treescan import TreeScanningGateRemovalPass
from bqskit.passes.synthesis.qsd import MGDPass, QSDPass
from bqskit.passes.processing.extract_diagonal import ExtractDiagonalPass
from bqskit.ir.circuit import Circuit

from bqskit.compiler.basepass import BasePass
from bqskit.ir.operation import Operation
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.mcry import MCRYGate
from bqskit.ir.gates.parameterized.mcrz import MCRZGate
from bqskit.ir.gates.constant import IdentityGate, HGate, ZGate
from bqskit.ir.gates.parameterized import RZGate
from bqskit.ir.gates.constant import CNOTGate
from bqskit.ir.location import CircuitLocation, CircuitLocationLike
from bqskit.ir.gates import CircuitGate
from bqskit.runtime import get_runtime
from scipy.linalg import svd, block_diag, eig
import numpy as np



_logger = logging.getLogger(__name__)

class FullBlockZXZPass(BasePass):
    """
    A pass performing a full Block ZXZ decomposition

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
        perform_extract: bool = True,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Perform the full Block ZXZ decomposition.

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
            perform_extract (bool): Whether or not to perform the diagonal
                extraction pass. (Default: True)
            instantiate_options (dict): The options to pass to the
                scanning gate removal pass. (Default: {})
        """

        self.start_from_left = start_from_left
        self.min_qudit_size = min_qudit_size
        instantiation_options = {"method":"qfactor"}
        instantiation_options.update(instantiate_options)
        self.perform_scan = perform_scan
        self.perform_extract = perform_extract
        if tree_depth > 0:
            self.scan = TreeScanningGateRemovalPass(
                start_from_left=start_from_left, 
                instantiate_options=instantiation_options, 
                tree_depth=tree_depth)
        else:
            self.scan = ScanningGateRemovalPass(
                start_from_left=start_from_left, 
                instantiate_options=instantiation_options)
        self.bzxz = BlockZXZPass(min_qudit_size=min_qudit_size)
        self.mgd = MGDPass(inverse = True)
        self.diag = ExtractDiagonalPass()

    async def run(self, circuit: Circuit, data: PassData) -> None:
        '''
        Perform succesive rounds of Block ZXZ decomposition until the
        circuit is fully decomposed with no VariableUnitaryGates larger 
        then `min_qudit_size`. 

        At the end, attempt to extrac the diagonal gate and 
        commute through the circuit to find optimal CNOT counts.
        '''
        passes = []
        start_num = max(x.num_qudits for x in circuit.operations())

        for _ in range(self.min_qudit_size, start_num):
            passes.append(self.bzxz)
            if self.perform_scan:
                passes.append(self.scan)
        
        if self.perform_extract:
            passes.append(self.diag)

        # Once we have commuted the diagonal gate, we can break down the 
        # multiplexed gates
        for _ in range(self.min_qudit_size, start_num):
            passes.append(self.mgd)
            if self.perform_scan:
                passes.append(self.scan)

        await Workflow(passes).run(circuit, data)

def left_shift(loc: CircuitLocationLike):
    return loc[1:] + loc[0:1]

class BlockZXZPass(BasePass):
    """
    A pass performing one round of decomposition from the Block ZXZ algorithm.

    References:
        Krol, Anna M., and Zaid Al-Ars. 
        "Highly Efficient Decomposition of n-Qubit Quantum Gates 
        Based on Block-ZXZ Decomposition." 
        arXiv preprint arXiv:2403.13692 (2024).
        https://arxiv.org/html/2403.13692v1
    """

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

    def initial_decompose(U: UnitaryMatrix) -> tuple[UnitaryMatrix, 
                                                     UnitaryMatrix, 
                                                     UnitaryMatrix, 
                                                     UnitaryMatrix]:
        '''
        This function decomposes the a given unitary U into 2 controlled
        unitaries (B, and C), 1 multiplexed unitary (A) and 2 Hadamard gates. 

        We return the A1 and A2 (sub-unitaries of the multipelexed unitary A),
        as well as the sub unitary B and C.

        By sub-unitary, we mean the unitary that will be applied to the bottom
        n - 1 qubits given the value of the top qubit 0. For a controlled 
        unitary, that sub-unitary is either the Identity (if qubit 0 is 0) or a
        unitary on n-1 qubits if qubit 0 is 1.

        This follows equations 5-9 in the paper.
        '''
        # To solve for A1, A2, and C, we must divide U into 4 submatrices.
        # The upper left block is X, the upper right block is Y, the lower left
        # block is U_21, and the lower right block is U_22.

        X = U[0:len(U) // 2, 0:len(U) // 2]
        Y = U[0:len(U) // 2, len(U) // 2:]
        U_21 = U[len(U) // 2:, 0:len(U) // 2]
        U_22 = U[len(U) // 2:, len(U) // 2:]

        # We let X = V_X @ s_X @ W_X where V_X and W_X are unitary and s_X is 
        # a diagonal matrix of singular values. We use the SVD to find these
        # matrices. We do the same decomposition for Y.
        V_X, s_X, W_XH = svd(X)
        V_Y, s_Y, W_YH = svd(Y)

        # We can then define S_X and S_Y as V_X @ s_X @ V_X^H and 
        # V_Y @ s_Y @ V_Y^H
        S_X = V_X @ np.diag(s_X) @ V_X.conj().T
        S_Y = V_Y @ np.diag(s_Y) @ V_Y.conj().T

        # We define U_X and U_Y as V_X^H @ U @ W_X and V_Y^H @ U @ W_Y
        U_X = V_X @ W_XH
        U_Y = V_Y @ W_YH

        # Now, to perform the decomposition as defined in Section 4.1
        # We can set C dagger as i(U_Y^H @ U_X)
        CH = 1j * U_Y.conj().T @ U_X

        # We then set A_1 as S_X @ U_X and A_2 as U_21 + U_22 @ (iU_Y^H @ U_X)
        A_1 = UnitaryMatrix((S_X + 1j * S_Y) @ U_X)
        A_2 = UnitaryMatrix(U_21 + U_22 @ (1j * U_Y.conj().T @ U_X))

        # Finally, we can set B as 2(A_1^H @ U) - I
        I = np.eye(len(U) / 2)
        B = UnitaryMatrix((2 * (A_1.conj().T @ X)) - I)

        return A_1, A_2, B, UnitaryMatrix(CH.conj().T)

    @staticmethod 
    def demultiplex(U_1: UnitaryMatrix, 
                    U_2: UnitaryMatrix) -> tuple[UnitaryMatrix, 
                                                 np.ndarray, 
                                                 UnitaryMatrix]:
        '''
        Demultiplex U that is block_diag(U_1, U_2) to
        2 Unitary Matrices V and W and a diagonal matrix D.

        The Diagonal Matrix D corresponds to a Multiplexed Rz gate acting on
        the most significant qubit.

        We return the Unitary Matrices V and W and the parameters for the
        corresponding MCRZ gate.

        '''
        # U can be decomposed into U = (I otimes V )(D otimes D†)(I otimes W )

        # We can find V,D^2 by performing an eigen decomposition of 
        # U_1 @ U_2†
        d2, V = eig(U_1 @ U_2.conj().T)
        d = np.sqrt(d2)
        D = np.diag(d)

        # We can then multiply to solve for W
        W = D @ V.conj().T @ U_2

        # We can use d to find the parameters for the MCRZ gate.
        # Note than because and Rz does exp(-i * theta / 2), we must 
        # multiply by -2.
        d_params = np.angle(d) * -2

        return UnitaryMatrix(V), d_params, UnitaryMatrix(W)

    @staticmethod
    def decompose_mpz_one_level(params, 
                                num_qudits, 
                                reverse=False, 
                                drop_last_cnot=False) -> Circuit:
        '''
        Decompose MPZ one level. 

        Args:
            params (np.ndarray): The parameters for the original MCRZ gate
            num_qudits (int): The number of qudits in the MCRZ gate
            reverse (bool): Whether to reverse the order of the gates (you can
            decompose the gates in either order to get the same result)
            drop_last_cnot (bool): Whether to drop the last CNOT gate (only 
            should be set to True if you are doing section 5.2 optimization)

        Returns:
            Circuit: The circuit that decomposes the MCRZ gate
        '''
        if (num_qudits >= 3):
            # Remove 1 qubit, last qubit is controlled
            new_gate = MCRZGate(num_qudits - 1, num_qudits-2)
        else:
            new_gate = RZGate()

        left_params, right_params =  MCRYGate.get_decomposition(params)
        circ = Circuit(num_qudits)
        new_gate_location = list(range(1, num_qudits))
        cx_location = (0, num_qudits - 1)

        ops = [Operation(new_gate, new_gate_location, left_params), 
            Operation(CNOTGate(), cx_location),
            Operation(new_gate, new_gate_location, right_params),
            Operation(CNOTGate(), cx_location)]
        
        if drop_last_cnot:
            ops.pop()
        
        if reverse:
            ops.reverse()

        for op in ops:
            circ.append(op)

        return circ

    @staticmethod
    def decompose_mpz_two_levels(params, num_qudits, reverse=False, remove_last_cnot=True) -> Circuit:
        '''
        We decompose a multiplexed RZ gate 2 levels deep. This allows you
        to remove the last CNOT gate in the context of the Block ZXZ 
        decomposition. This is shown in section 5.2 of the paper.

        Args:
            params (np.ndarray): The parameters for the original MCRZ gate
            num_qudits (int): The number of qudits in the MCRZ gate
            reverse (bool): Whether to reverse the order of the gates (you can
            decompose the gates in either order to get the same result)
            drop_last_cnot (bool): Whether to drop the last CNOT gate (only 
            should be set to True if you are doing section 5.2 optimization)
        
        Returns:
            Circuit: The circuit that decomposes the MCRZ gate
        '''
        # Get params for first decomposition of the MCRZ gate
        left_params, right_params =  MCRYGate.get_decomposition(params)


        # Decompose the MCRZ gate into 2 MCRZ gates, dropping the last CNOT
        # Also Reverse the circuit for the right side in order to do the 
        # optimization in section 5.2
        circ_left = BlockZXZPass.decompose_mpz_one_level(left_params, 
                                                         num_qudits - 1, 
                                                         reverse, 
                                                         drop_last_cnot=True)
        
        circ_right = BlockZXZPass.decompose_mpz_one_level(right_params, 
                                                          num_qudits - 1, 
                                                          not reverse, 
                                                          drop_last_cnot=True)

        # Now, construct the circuit.
        # This will generate the original MCRZ gate with the target qubit
        # set as qubit num_qudits - 1
        circ = Circuit(num_qudits)
        cx_location_big = (0, num_qudits - 1)

        ops = [circ_left,
            Operation(CNOTGate(), cx_location_big),
            circ_right,
            Operation(CNOTGate(), cx_location_big)]
        
        # We can remove the last CNOT gate as per section 5.2
        if remove_last_cnot:
            ops.pop()
        
        if reverse:
            ops.reverse()

        for op in ops:
            if isinstance(op, Operation):
                circ.append(op)
            else:
                circ.append_circuit(op, list(range(1, num_qudits)))

        return circ


    @staticmethod
    def zxz(orig_u: UnitaryMatrix) -> Circuit:
        '''
        Return the circuit that is generated from one levl of 
        Block ZXZ decomposition. 
        '''

        # First calculate the A, B, and C matrices for the initial decomp
        A_1, A_2, B, C = BlockZXZPass.initial_decompose(orig_u)

        # Now decompose thee multiplexed A gate and the controlled C gate
        I = IdentityGate(orig_u.num_qudits - 1).get_unitary()
        VA, AZ_params, WA = BlockZXZPass.demultiplex(A_1, A_2)
        VC, CZ_params, WC = BlockZXZPass.demultiplex(I, C) 

        # Now calculate optimized B_tilde (Section 5.2) and decompose
        # We merge in WA and VC into B and then merge in the additional 
        # CZ gates from the optimization
        small_I = IdentityGate(orig_u.num_qudits - 2).get_unitary()
        Z = ZGate(1).get_unitary()
        B_tilde_1 = WA @ VC
        B_tilde_2 = np.kron(Z, small_I) @ WA @ B @ VC @ np.kron(Z, small_I)
        VB, BZ_params, WB = BlockZXZPass.demultiplex(B_tilde_1, B_tilde_2)

        # Define circuit locations
        # We let the target qubit be qubit 0 (top qubit in diagram)
        # The select qubits are the remaining qubits
        controlled_qubit = 0
        select_qubits = list(range(1, orig_u.num_qudits))
        all_qubits = list(range(0, orig_u.num_qudits))

        # z_gate_circ = BlockZXZPass.decompose_mpz_two_levels(BZ_params, orig_u.num_qudits, False, False)
        # assert(np.allclose(z_gate_circ.get_unitary(), z_gate.get_unitary(BZ_params)))

        # Construct Circuit
        circ = Circuit(orig_u.num_qudits)

        # Add WC gate
        wc_gate, wc_params = QSDPass.create_unitary_gate(WC)
        circ.append_gate(wc_gate, CircuitLocation(select_qubits), wc_params)


        # Add decomposed MCRZ gate circuit. 
        # Since the MCRZ gate circuit sets the target qubit as qubit
        # num_qudits - 1, we must shift the qubits to the left
        shifted_qubits = left_shift(all_qubits)
        circ.append_circuit(
            BlockZXZPass.decompose_mpz_two_levels(CZ_params, orig_u.num_qudits), 
            shifted_qubits)


        # Add the decomposed B-tilde gates WB and a Hadamard
        combo_1_gate, combo_1_params = QSDPass.create_unitary_gate(WB)
        circ.append_gate(combo_1_gate, CircuitLocation(select_qubits), 
                         combo_1_params)
        circ.append_gate(HGate(), CircuitLocation((controlled_qubit, )))

        # The central MCRZ_gate. We set the target to the controlled qubit,
        # so there is no need to shift
        z_gate = MCRZGate(len(all_qubits), 0)
        circ.append_gate(z_gate, CircuitLocation(all_qubits), BZ_params)

        # Now add the decomposed B-tilde gates VB and a Hadamard
        combo_2_gate, combo_2_params = QSDPass.create_unitary_gate(VB)
        circ.append_gate(combo_2_gate, CircuitLocation(select_qubits), 
                         combo_2_params)
        circ.append_gate(HGate(), CircuitLocation((controlled_qubit, )))


        # Add the decomposed MCRZ gate circuit again on shifted qubits
        circ.append_circuit(
            BlockZXZPass.decompose_mpz_two_levels(AZ_params, 
                                                  orig_u.num_qudits, 
                                                  True), 
                                                  shifted_qubits)
    
        va_gate, va_params = QSDPass.create_unitary_gate(VA)
        circ.append_gate(va_gate, CircuitLocation(select_qubits), va_params)


        # assert np.allclose(orig_u, circ.get_unitary())
        return circ

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform a single level of the Block ZXZ decomposition."""
        unitaries, pts, locations = QSDPass.get_variable_unitary_pts(
            circuit, self.min_qudit_size
        )

        if len(unitaries) > 0:
            circs = await get_runtime().map(BlockZXZPass.zxz, unitaries)
            # Do bulk replace
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [Operation(x, locations[i], x._circuit.params) 
                        for i,x in enumerate(circ_gates)]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        circuit.unfold_all()