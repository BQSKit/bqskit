"""This module implements functions for translating to and from QuTiP."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qutip import QubitCircuit

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def qutip_to_bqskit(qc: QubitCircuit) -> Circuit:
    """Convert QuTiP's QubitCircuit `qc` to a BQSKit Circuit."""
    try:
        from qutip.qip.qasm import circuit_to_qasm_str
    except ImportError as e:
        raise ImportError(
            '\n\nUnable to import qutip package.\n'
            'Please ensure that it is installed with the following command:\n'
            '\tpip install qutip\n',
        ) from e

    circuit = OPENQASM2Language().decode(circuit_to_qasm_str(qc))
    return circuit


def bqskit_to_qutip(circuit: Circuit) -> QubitCircuit:
    """Convert a BQSKit Circuit to QuTiP's QubitCircuit."""
    try:
        from qutip.qip.qasm import read_qasm
    except ImportError as e:
        raise ImportError(
            '\n\nUnable to import qutip package.\n'
            'Please ensure that it is installed with the following command:\n'
            '\tpip install qutip\n',
        ) from e

    return read_qasm(OPENQASM2Language().encode(circuit), strmode=True)
