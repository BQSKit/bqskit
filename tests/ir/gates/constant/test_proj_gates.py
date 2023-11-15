from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.constant.clock import ClockGate
from bqskit.ir.gates.constant.csum import CSUMGate
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.pd import PDGate
from bqskit.ir.gates.constant.shift import ShiftGate
from bqskit.ir.gates.constant.subswap import SubSwapGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.z import ZGate


def build_proj_matrix(i: int, j: int, d: int) -> npt.NDArray[np.complex128]:
    """Builds the |i><j| projection matrix with dimension `d`."""
    mat = np.zeros([d, d], dtype=np.complex128)
    mat[i, j] = 1
    return mat


def test_csum_gate_proj() -> None:
    for radix in [2, 3, 4, 5]:
        csum = CSUMGate(radix)
        csum_unitary = csum.get_unitary()

        expected = np.zeros((radix ** 2, radix ** 2), dtype=np.complex128)
        for i in range(radix):
            for j in range(radix):
                ket = i * radix + ((i + j) % radix)
                bra = i * radix + j
                expected += build_proj_matrix(ket, bra, radix ** 2)

        assert np.allclose(csum_unitary, expected)


def test_csum_gate_is_cx_with_qubits() -> None:
    csum = CSUMGate(2)
    cx = CXGate()
    assert np.allclose(csum.get_unitary(), cx.get_unitary())


def test_clock_gate_proj() -> None:
    for radix in [2, 3, 4, 5]:
        clock = ClockGate(radix)
        clock_unitary = clock.get_unitary()

        expected = np.zeros((radix, radix), dtype=np.complex128)
        for i in range(radix):
            proj = build_proj_matrix(i, i, radix)
            expected += np.exp(2j * np.pi * i / radix) * proj

        assert np.allclose(clock_unitary, expected)


def test_clock_gate_is_z_with_qubits() -> None:
    clock = ClockGate(2)
    z = ZGate()
    assert np.allclose(clock.get_unitary(), z.get_unitary())


def test_h_gate_proj() -> None:
    for radix in [2, 3, 4, 5]:
        h = HGate(radix)
        h_unitary = h.get_unitary()

        expected = np.zeros((radix, radix), dtype=np.complex128)
        for i in range(radix):
            for j in range(radix):
                proj = build_proj_matrix(i, j, radix)
                expected += np.exp(2j * np.pi * i * j / radix) * proj

        expected /= np.sqrt(radix)

        assert np.allclose(h_unitary, expected)


def test_h_gate_is_qubit_hadamard() -> None:
    h = HGate(2)
    h_unitary = h.get_unitary()

    expected = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
        ],
        dtype=np.complex128,
    )
    assert np.allclose(h_unitary, expected)


def test_pd_gate_proj() -> None:
    for radix in [2, 3, 4, 5]:
        for i in range(radix):
            pd = PDGate(i, radix)
            pd_unitary = pd.get_unitary()

            omega = np.exp(2j * np.pi * i / radix)
            expected = np.zeros((radix, radix), dtype=np.complex128)
            for j in range(radix):
                proj = build_proj_matrix(j, j, radix)
                expected += ((-omega ** 2) ** (1 if i == j else 0)) * proj

            assert np.allclose(pd_unitary, expected)


def test_shift_gate_proj() -> None:
    for radix in [2, 3, 4, 5]:
        shift = ShiftGate(radix)
        shift_unitary = shift.get_unitary()

        expected = np.zeros((radix, radix), dtype=np.complex128)
        for i in range(radix):
            expected += build_proj_matrix((i + 1) % radix, i, radix)

        assert np.allclose(shift_unitary, expected)


def test_shift_gate_is_x_with_qubits() -> None:
    shift = ShiftGate(2)
    x = XGate()
    assert np.allclose(shift.get_unitary(), x.get_unitary())


def test_sub_swap_proj() -> None:
    for radix in [2, 3, 4, 5]:
        for i in range(radix):
            for j in range(radix):
                subswap = SubSwapGate(radix, f'{i},{j};{j},{i}')
                subswap_unitary = subswap.get_unitary()

                dim = (radix ** 2, radix ** 2)
                expected = np.zeros(dim, dtype=np.complex128)
                for k in range(radix):
                    for l in range(radix):
                        ind = k * radix + l
                        inv = l * radix + k
                        if (k == i and l == j) or (k == j and l == i):
                            expected += build_proj_matrix(ind, inv, radix ** 2)

                        else:
                            expected += build_proj_matrix(ind, ind, radix ** 2)

                assert np.allclose(subswap_unitary, expected)


def test_sub_swap_10_11_qubit_is_cnot() -> None:
    subswap = SubSwapGate(2, '1,0;1,1')
    cnot = CXGate()
    assert np.allclose(subswap.get_unitary(), cnot.get_unitary())


def test_sub_swap_01_10_qubit_is_swap() -> None:
    subswap = SubSwapGate(2, '0,1;1,0')
    swap = SwapGate()
    assert np.allclose(subswap.get_unitary(), swap.get_unitary())
