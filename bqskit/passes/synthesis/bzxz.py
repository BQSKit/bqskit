"""
This module implements the Full Block ZXZ Decomposition.

Additionally, it defines the Block ZXZ Decomposition for a single pass.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.linalg import eig
from scipy.linalg import svd

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant import HGate
from bqskit.ir.gates.constant import IdentityGate
from bqskit.ir.gates.constant import ZGate
from bqskit.ir.gates.parameterized.mcrz import MCRZGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.passes.processing.extract_diagonal import ExtractDiagonalPass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.treescan import TreeScanningGateRemovalPass
from bqskit.passes.synthesis.qsd import MGDPass
from bqskit.passes.synthesis.qsd import QSDPass
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.runtime import get_runtime


_logger = logging.getLogger(__name__)


class FullBlockZXZPass(BasePass):
    """
    A pass performing a full Block ZXZ decomposition.

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
        instantiation_options = {'method': 'qfactor'}
        instantiation_options.update(instantiate_options)
        self.perform_scan = perform_scan
        self.perform_extract = perform_extract
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
        self.bzxz = BlockZXZPass(min_qudit_size=min_qudit_size)
        self.mgd = MGDPass()
        self.diag = ExtractDiagonalPass(qudit_size=min_qudit_size)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Perform succesive rounds of Block ZXZ decomposition until the circuit is
        fully decomposed with no VariableUnitaryGates larger then
        `min_qudit_size`.

        At the end, attempt to extrac the diagonal gate and commute through the
        circuit to find optimal CNOT counts.
        """
        passes: list[BasePass] = []
        start_num = max(x.num_qudits for x in circuit.operations())

        for _ in range(self.min_qudit_size, start_num):
            passes.append(self.bzxz)
            if self.perform_scan:
                passes.append(self.scan)

        if self.perform_extract:
            passes.append(self.diag)

        # Once we have commuted the diagonal gate, we can break down the
        # multiplexed gates
        for _ in range(1, start_num):
            passes.append(self.mgd)
            if self.perform_scan:
                passes.append(self.scan)

        await Workflow(passes).run(circuit, data)


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

    @staticmethod
    def initial_decompose(U: UnitaryMatrix) -> tuple[
        UnitaryMatrix,
        UnitaryMatrix,
        UnitaryMatrix,
        UnitaryMatrix,
    ]:
        """
        This function decomposes the a given unitary U into 2 controlled
        unitaries (B, and C), 1 multiplexed unitary (A) and 2 Hadamard gates.

        We return the A1 and A2 (sub-unitaries of the multipelexed unitary A),
        as well as the sub unitary B and C.

        By sub-unitary, we mean the unitary that will be applied to the bottom
        n - 1 qubits given the value of the top qubit 0. For a controlled
        unitary, that sub-unitary is either the Identity (if qubit 0 is 0) or a
        unitary on n-1 qubits if qubit 0 is 1.

        This follows equations 5-9 in the paper.
        """
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
        I = np.eye(len(U) // 2)
        B = UnitaryMatrix((2 * (A_1.conj().T @ X)) - I)

        return A_1, A_2, B, UnitaryMatrix(CH.conj().T)

    @staticmethod
    def demultiplex(
        U_1: UnitaryMatrix,
        U_2: UnitaryMatrix,
    ) -> tuple[
        UnitaryMatrix,
        RealVector,
        UnitaryMatrix,
    ]:
        """
        Demultiplex U that is block_diag(U_1, U_2) to 2 Unitary Matrices V and W
        and a diagonal matrix D.

        The Diagonal Matrix D corresponds to a Multiplexed Rz gate acting on the
        most significant qubit.

        We return the Unitary Matrices V and W and the parameters for the
        corresponding MCRZ gate.
        """
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
        d_params: list[float] = list(np.angle(d) * -2)

        return UnitaryMatrix(V), d_params, UnitaryMatrix(W)

    @staticmethod
    def zxz(orig_u: UnitaryMatrix) -> Circuit:
        """Return the circuit that is generated from one levl of Block ZXZ
        decomposition."""

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
        B_tilde_1 = UnitaryMatrix(WA @ VC)
        B_tilde_2 = UnitaryMatrix(
            np.kron(Z, small_I) @
            WA @ B @ VC @ np.kron(Z, small_I),
        )
        VB, BZ_params, WB = BlockZXZPass.demultiplex(B_tilde_1, B_tilde_2)

        # Define circuit locations
        # We let the target qubit be qubit 0 (top qubit in diagram)
        # The select qubits are the remaining qubits
        controlled_qubit = 0
        select_qubits = list(range(1, orig_u.num_qudits))
        all_qubits = list(range(0, orig_u.num_qudits))

        # Construct Circuit
        circ = Circuit(orig_u.num_qudits)

        # Add WC gate
        wc_gate, wc_params = QSDPass.create_unitary_gate(WC)
        circ.append_gate(wc_gate, CircuitLocation(select_qubits), wc_params)

        # Add decomposed MCRZ gate circuit.
        # Since the MCRZ gate circuit sets the target qubit as qubit
        # num_qudits - 1, we must shift the qubits to the left
        shifted_qubits = all_qubits[1:] + all_qubits[0:1]
        circ.append_circuit(
            MGDPass.decompose_mpx_two_levels(
                decompose_ry=False,
                params=CZ_params,
                num_qudits=orig_u.num_qudits,
                drop_last_cnot=True,
            ),
            shifted_qubits,
        )

        # Add the decomposed B-tilde gates WB and a Hadamard
        combo_1_gate, combo_1_params = QSDPass.create_unitary_gate(WB)
        circ.append_gate(
            combo_1_gate, CircuitLocation(select_qubits),
            combo_1_params,
        )
        circ.append_gate(HGate(), CircuitLocation((controlled_qubit,)))

        # The central MCRZ_gate. We set the target to the controlled qubit,
        # so there is no need to shift
        z_gate = MCRZGate(len(all_qubits), 0)
        circ.append_gate(z_gate, CircuitLocation(all_qubits), BZ_params)

        # Now add the decomposed B-tilde gates VB and a Hadamard
        combo_2_gate, combo_2_params = QSDPass.create_unitary_gate(VB)
        circ.append_gate(
            combo_2_gate, CircuitLocation(select_qubits),
            combo_2_params,
        )
        circ.append_gate(HGate(), CircuitLocation((controlled_qubit,)))

        # Add the decomposed MCRZ gate circuit again on shifted qubits
        circ.append_circuit(
            MGDPass.decompose_mpx_two_levels(
                decompose_ry=False,
                params=AZ_params,
                num_qudits=orig_u.num_qudits,
                reverse=True,
                drop_last_cnot=True,
            ),
            shifted_qubits,
        )

        va_gate, va_params = QSDPass.create_unitary_gate(VA)
        circ.append_gate(va_gate, CircuitLocation(select_qubits), va_params)

        # assert np.allclose(orig_u, circ.get_unitary())
        return circ

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform a single level of the Block ZXZ decomposition."""
        unitaries, pts, locations = QSDPass.get_variable_unitary_pts(
            circuit, self.min_qudit_size,
        )

        if len(unitaries) > 0:
            circs = await get_runtime().map(BlockZXZPass.zxz, unitaries)
            # Do bulk replace
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [
                Operation(x, locations[i], x._circuit.params)
                for i, x in enumerate(circ_gates)
            ]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        circuit.unfold_all()
